import sys
import os
import asyncio
import m3u8
from scrapling.fetchers import AsyncFetcher, FetcherSession

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QTextEdit, QFileDialog, QGroupBox,
)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QFont, QTextCursor

def get_app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = get_app_dir()


def resolve_url(uri, playlist, m3u8_url):
    if uri.startswith("http"):
        return uri
    if playlist.base_uri:
        return playlist.base_uri + uri
    return os.path.dirname(m3u8_url) + "/" + uri


async def download_one(url, semaphore, referer):
    async with semaphore:
        resp = await AsyncFetcher.get(
            url, stealthy_headers=False, headers={"referer": referer}, timeout=120
        )
        if resp.status != 200:
            raise RuntimeError(f"状态码 {resp.status}")
        return resp.body


async def download_all_segments(segments, playlist, m3u8_url, dl_dir, progress_cb):
    semaphore = asyncio.Semaphore(30)
    total = len(segments)
    done_count = 0
    lock = asyncio.Lock()

    async def worker(idx, seg):
        nonlocal done_count
        seg_url = resolve_url(seg.uri, playlist, m3u8_url)
        try:
            data = await download_one(seg_url, semaphore, m3u8_url)
            filename = f"seg_{idx:05d}.mp4"
            with open(os.path.join(dl_dir, filename), "wb") as f:
                f.write(data)
        except Exception as e:
            async with lock:
                done_count += 1
                progress_cb(f"!! 分片 {idx} 失败: {e}")
            return

        async with lock:
            done_count += 1
            if done_count % 20 == 0 or done_count == total:
                size_mb = sum(
                    os.path.getsize(os.path.join(dl_dir, f))
                    for f in os.listdir(dl_dir)
                    if f.startswith("seg_")
                ) / 1024 / 1024
                progress_cb(f"进度: {done_count}/{total}  {size_mb:.1f} MB")

    tasks = [worker(i, seg) for i, seg in enumerate(segments, 1)]
    await asyncio.gather(*tasks)


async def download_video_async(m3u8_url, output_path, progress_cb):
    progress_cb("[1/4] 获取 m3u8 播放列表")

    with FetcherSession(impersonate="chrome") as session:
        response = session.get(m3u8_url, stealthy_headers=False)
        if response.status != 200:
            raise RuntimeError(f"请求失败，状态码: {response.status}")
        m3u8_text = response.body.decode("utf-8")

    progress_cb("[2/4] 解析播放列表")
    playlist = m3u8.loads(m3u8_text, uri=m3u8_url)
    progress_cb(f"     共 {len(playlist.segments)} 个分片")

    if playlist.segment_map:
        for sm in playlist.segment_map:
            progress_cb(f"     EXT-X-MAP: {sm.uri}")

    dl_dir = os.path.join(OUTPUT_DIR, "video_dl")
    os.makedirs(dl_dir, exist_ok=True)
    progress_cb(f"     下载目录: {dl_dir}")

    if playlist.segment_map:
        progress_cb("     下载初始化片段...")
        for sm in playlist.segment_map:
            sm_url = resolve_url(sm.uri, playlist, m3u8_url)
            sm_resp = await AsyncFetcher.get(
                sm_url, stealthy_headers=False,
                headers={"referer": m3u8_url}, timeout=30
            )
            if sm_resp.status != 200:
                raise RuntimeError(f"init 下载失败，状态码: {sm_resp.status}")
            with open(os.path.join(dl_dir, "init.mp4"), "wb") as f:
                f.write(sm_resp.body)
            progress_cb(f"        {len(sm_resp.body) / 1024:.1f} KB")

    progress_cb(f"[3/4] 下载分片 ({len(playlist.segments)} 个, 并发 30)")
    await download_all_segments(playlist.segments, playlist, m3u8_url, dl_dir, progress_cb)

    progress_cb("[4/4] 合并为 MP4")
    total = 0
    with open(output_path, "wb") as out_f:
        init_path = os.path.join(dl_dir, "init.mp4")
        if os.path.exists(init_path):
            with open(init_path, "rb") as f:
                out_f.write(f.read())
            progress_cb("     写入 init.mp4")

        for i in range(1, len(playlist.segments) + 1):
            seg_path = os.path.join(dl_dir, f"seg_{i:05d}.mp4")
            if os.path.exists(seg_path):
                with open(seg_path, "rb") as f:
                    out_f.write(f.read())
                total += 1

    if os.path.basename(dl_dir) == "video_dl" and os.path.isdir(dl_dir):
        for f in os.listdir(dl_dir):
            fpath = os.path.join(dl_dir, f)
            if os.path.isfile(fpath):
                os.remove(fpath)
        os.rmdir(dl_dir)
        progress_cb("     已清理临时分片文件夹")

    final_size = os.path.getsize(output_path)
    progress_cb(f"\n完成! 合并 {total} 个分片，已清理碎片文件夹")
    progress_cb(f"文件: {output_path}")
    progress_cb(f"大小: {final_size / 1024 / 1024:.1f} MB")


class DownloadThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, m3u8_url, output_path):
        super().__init__()
        self.m3u8_url = m3u8_url
        self.output_path = output_path

    def run(self):
        try:
            asyncio.run(download_video_async(
                self.m3u8_url,
                self.output_path,
                progress_cb=self.log_signal.emit
            ))
            self.finished_signal.emit(True, "下载完成!")
        except Exception as e:
            self.finished_signal.emit(False, str(e))


class M3U8DownloaderGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.download_thread = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("M3U8 视频下载器")
        self.setMinimumSize(700, 500)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)

        url_group = QGroupBox("M3U8 地址")
        url_layout = QHBoxLayout(url_group)
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("请输入 m3u8 视频 URL...")
        url_layout.addWidget(self.url_input)
        main_layout.addWidget(url_group)

        save_group = QGroupBox("保存位置")
        save_layout = QHBoxLayout(save_group)
        self.save_path_input = QLineEdit()
        self.save_path_input.setPlaceholderText("选择视频文件保存路径...")
        self.save_path_input.setText(os.path.join(OUTPUT_DIR, "video.mp4"))
        save_layout.addWidget(self.save_path_input)
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.clicked.connect(self.browse_save_path)
        save_layout.addWidget(self.browse_btn)
        main_layout.addWidget(save_group)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.start_btn = QPushButton("开始下载")
        self.start_btn.setMinimumSize(150, 40)
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-size: 16px; font-weight: bold; border-radius: 5px; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:disabled { background-color: #cccccc; }"
        )
        self.start_btn.clicked.connect(self.start_download)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

        log_group = QGroupBox("下载日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        self.log_text.setStyleSheet("QTextEdit { background-color: #1e1e1e; color: #d4d4d4; }")
        log_layout.addWidget(self.log_text)
        main_layout.addWidget(log_group)

    def browse_save_path(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "选择保存位置",
            self.save_path_input.text(),
            "MP4 视频 (*.mp4);;所有文件 (*.*)"
        )
        if file_path:
            self.save_path_input.setText(file_path)

    def start_download(self):
        m3u8_url = self.url_input.text().strip()
        output_path = self.save_path_input.text().strip()

        if not m3u8_url:
            self.append_log("错误: 请输入 m3u8 URL")
            return
        if not output_path:
            self.append_log("错误: 请选择保存路径")
            return

        save_dir = os.path.dirname(os.path.abspath(output_path))
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir, exist_ok=True)
            except Exception as e:
                self.append_log(f"错误: 无法创建保存目录 - {e}")
                return

        self.start_btn.setEnabled(False)
        self.url_input.setEnabled(False)
        self.save_path_input.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.log_text.clear()

        self.append_log(f"开始下载...")
        self.append_log(f"URL: {m3u8_url}")
        self.append_log(f"保存至: {output_path}")
        self.append_log("-" * 50)

        self.download_thread = DownloadThread(m3u8_url, output_path)
        self.download_thread.log_signal.connect(self.append_log)
        self.download_thread.finished_signal.connect(self.on_download_finished)
        self.download_thread.start()

    def append_log(self, msg):
        self.log_text.append(msg)
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)

    def on_download_finished(self, success, message):
        self.append_log("-" * 50)
        if success:
            self.append_log(f"✓ {message}")
        else:
            self.append_log(f"✗ 下载失败: {message}")

        self.start_btn.setEnabled(True)
        self.url_input.setEnabled(True)
        self.save_path_input.setEnabled(True)
        self.browse_btn.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = M3U8DownloaderGUI()
    window.show()
    sys.exit(app.exec_())