import asyncio
import os
import m3u8
from scrapling.fetchers import AsyncFetcher, FetcherSession

M3U8_URL = ""
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_url(uri, playlist, m3u8_url):
    if uri.startswith("http"):
        return uri
    if playlist.base_uri:
        return playlist.base_uri + uri
    return os.path.dirname(m3u8_url) + "/" + uri


async def download_one(url, semaphore):
    async with semaphore:
        resp = await AsyncFetcher.get(
            url, stealthy_headers=False, headers={"referer": M3U8_URL}, timeout=120
        )
        if resp.status != 200:
            raise RuntimeError(f"状态码 {resp.status}")
        return resp.body


async def download_all_segments(segments, playlist, m3u8_url, dl_dir):
    semaphore = asyncio.Semaphore(30)
    total = len(segments)
    done_count = 0
    lock = asyncio.Lock()

    async def worker(idx, seg):
        nonlocal done_count
        seg_url = resolve_url(seg.uri, playlist, m3u8_url)
        try:
            data = await download_one(seg_url, semaphore)
            filename = f"seg_{idx:05d}.mp4"
            with open(os.path.join(dl_dir, filename), "wb") as f:
                f.write(data)
        except Exception as e:
            async with lock:
                done_count += 1
                print(f"       !! 分片 {idx} 失败: {e}")
            return

        async with lock:
            done_count += 1
            if done_count % 20 == 0 or done_count == total:
                size_mb = sum(
                    os.path.getsize(os.path.join(dl_dir, f))
                    for f in os.listdir(dl_dir)
                    if f.startswith("seg_")
                ) / 1024 / 1024
                print(f"       进度: {done_count}/{total}  {size_mb:.1f} MB")

    tasks = [worker(i, seg) for i, seg in enumerate(segments, 1)]
    await asyncio.gather(*tasks)


async def download_video_async(m3u8_url, output_path):
    print(f"[1/4] 获取 m3u8 播放列表")

    with FetcherSession(impersonate="chrome") as session:
        response = session.get(m3u8_url, stealthy_headers=False)
        if response.status != 200:
            raise RuntimeError(f"请求失败，状态码: {response.status}")
        m3u8_text = response.body.decode("utf-8")

    print(f"[2/4] 解析播放列表")
    playlist = m3u8.loads(m3u8_text, uri=m3u8_url)

    print(f"       共 {len(playlist.segments)} 个分片")

    if playlist.segment_map:
        for sm in playlist.segment_map:
            print(f"       EXT-X-MAP: {sm.uri}")

    dl_dir = os.path.join(OUTPUT_DIR, "video_2160p")
    os.makedirs(dl_dir, exist_ok=True)
    print(f"       下载目录: {dl_dir}")

    if playlist.segment_map:
        print(f"       下载初始化片段...")
        for sm in playlist.segment_map:
            sm_url = resolve_url(sm.uri, playlist, m3u8_url)
            sm_resp = await AsyncFetcher.get(sm_url, stealthy_headers=False, headers={"referer": m3u8_url}, timeout=30)
            if sm_resp.status != 200:
                raise RuntimeError(f"init 下载失败，状态码: {sm_resp.status}")
            with open(os.path.join(dl_dir, "init.mp4"), "wb") as f:
                f.write(sm_resp.body)
            print(f"          {len(sm_resp.body) / 1024:.1f} KB")

    print(f"[3/4] 下载分片 ({len(playlist.segments)} 个, 并发 30)")
    await download_all_segments(playlist.segments, playlist, m3u8_url, dl_dir)

    print(f"[4/4] 合并为 MP4")
    total = 0
    with open(output_path, "wb") as out_f:
        init_path = os.path.join(dl_dir, "init.mp4")
        if os.path.exists(init_path):
            with open(init_path, "rb") as f:
                out_f.write(f.read())
            print(f"       写入 init.mp4")

        for i in range(1, len(playlist.segments) + 1):
            seg_path = os.path.join(dl_dir, f"seg_{i:05d}.mp4")
            if os.path.exists(seg_path):
                with open(seg_path, "rb") as f:
                    out_f.write(f.read())
                total += 1

    for f in os.listdir(dl_dir):
        os.remove(os.path.join(dl_dir, f))
    os.rmdir(dl_dir)

    final_size = os.path.getsize(output_path)
    print(f"\n完成! 合并 {total} 个分片，已清理碎片文件夹")
    print(f"文件: {output_path}")
    print(f"大小: {final_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    output_file = os.path.join(OUTPUT_DIR, "s5.mp4")
    asyncio.run(download_video_async(M3U8_URL, output_file))