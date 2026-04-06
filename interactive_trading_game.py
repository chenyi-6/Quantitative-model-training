# 导入pygame库，用于图形界面和游戏开发
import pygame
import sys
import pandas as pd
import numpy as np
import os
import random
import datetime


class KLineViewer:
    """
    K线行情模拟查看器
    包含K线、成交量、布林轨道和交易系统
    """

    # 窗口尺寸
    WIDTH = 1200
    HEIGHT = 720  # 增加高度以容纳交易信息
    CHART_HEIGHT = 400

    def __init__(self, data_path, max_steps=5000):
        """
        初始化K线查看器

        Args:
            data_path (str): 包含历史交易数据的CSV文件路径
            max_steps (int): 最大步数，达到该步数后强制平仓并结束
        """
        # 加载数据
        self.data = self.load_futures_data(data_path)

        # 计算布林带
        self._calculate_bollinger_bands()

        # 初始化Pygame
        pygame.init()
        # 添加 RESIZABLE 标志使窗口可以调整大小
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.RESIZABLE)
        # 设置命名
        pygame.display.set_caption("K线交易模拟器")

        self.clock = pygame.time.Clock()

        self.font = pygame.font.Font("C:/Windows/Fonts/simsun.ttc", 17)
        self.small_font = pygame.font.Font("C:/Windows/Fonts/simsun.ttc", 20)

        # 图表参数
        self.chart_offset = 0
        self.chart_scale = 1.0
        # 当前显示的数据范围，初始化为随机截取
        self._random_slice()

        # 自动播放相关参数
        self.play_speed = 30  # 每秒刷新帧数#数值越大速度越快
        self.last_update_time = 0

        # 最大步数
        self.max_steps = max_steps
        self.current_step = 0

        # 交易系统初始化
        self._init_trading_system()

    def _init_trading_system(self):
        """初始化交易系统"""
        # 账户信息
        self.initial_capital = 100000  # 初始资金10万
        self.cash = self.initial_capital
        self.total_value = self.initial_capital

        # 持仓信息
        self.position = 0  # 持仓数量，正数为多头，负数为空头
        self.avg_cost = 0  # 平均成本
        self.position_value = 0  # 持仓市值

        # 交易记录
        self.trade_history = []

        # 当前K线索引
        self.current_kline_index = self.end_index - 1

        # 交易参数
        self.trade_volume = 1  # 每次交易手数

        # 浮动盈亏
        self.floating_pnl = 0

        # 持仓状态
        self.position_type = "空仓"  # 空仓/多头/空头

        # 重置步数
        self.current_step = 0

    def load_futures_data(self, filepath):
        """
        加载期货历史数据

        Args:
            filepath (str): 数据文件路径

        Returns:
            pd.DataFrame: 加载的数据
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

        data = pd.read_csv(filepath, encoding='utf-8')
        data['date'] = pd.to_datetime(data['datetime'])
        data = data.sort_values('date')
        data = data.reset_index(drop=True)
        return data

    def _calculate_bollinger_bands(self, window=27, num_std=2):
        """
        计算布林带指标
        """
        # 使用滞后计算确保不会出现前瞻偏差
        self.data['ma'] = self.data['close'].rolling(window=window).mean()
        self.data['std'] = self.data['close'].rolling(window=window).std()
        self.data['upper_band'] = self.data['ma'] + (self.data['std'] * num_std)
        self.data['lower_band'] = self.data['ma'] - (self.data['std'] * num_std)

    def _random_slice(self):
        """
        随机截取行情数据
        """
        max_start = len(self.data) - 1000
        self.start_index = random.randint(0, max_start)
        self.end_index = min(len(self.data), self.start_index + 1000)
        self.current_kline_index = self.end_index - 1

    def run(self):
        """
        运行K线查看器主循环
        """
        running = True
        while running:
            current_time = pygame.time.get_ticks()

            # 处理自动播放逻辑

            self._scroll(1)  # 自动向右滚动一根K线
            self.last_update_time = current_time

            # 处理事件
            for event in pygame.event.get():
                self._handle_event(event)

            # 更新浮动盈亏
            self._update_floating_pnl()

            # 渲染画面
            self.render()

            # 控制帧率
            self.clock.tick(self.play_speed)

        # 关闭Pygame
        pygame.quit()
        sys.exit()

    def _handle_events(self):
        """处理事件队列中的所有事件"""
        for event in pygame.event.get():
            self._handle_event(event)

    def _handle_event(self, event):
        """处理单个事件
        
        Args:
            event: pygame事件对象
        """
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            elif event.key == pygame.K_UP:
                # 缩放图表
                self.chart_scale *= 1.3
            elif event.key == pygame.K_DOWN:
                # 缩放图表
                self.chart_scale /= 1.3
            elif event.key == pygame.K_b:  # B键买入
                self.buy_action()
            elif event.key == pygame.K_s:  # S键卖出
                self.sell_action()

        elif event.type == pygame.VIDEORESIZE:
            self._handle_resize(event)

    def _scroll(self, amount):
        """
        滚动图表
        """
        self.start_index = max(0, min(len(self.data) - 100, self.start_index + amount))
        self.end_index = min(len(self.data), self.start_index + 100)
        self.current_kline_index = self.end_index - 1
        self.current_step += 1

    def render(self):
        """
        渲染画面
        """
        # 填充背景
        self.screen.fill((20, 20, 40))

        # 绘制K线图
        self._draw_chart()

        # 绘制信息面板
        self._draw_info_panel()

        # 绘制交易面板
        self._draw_trading_panel()

        # 更新显示
        pygame.display.flip()

    def _handle_resize(self, event):
        """
        处理窗口大小调整事件
        """
        self.WIDTH, self.HEIGHT = event.size
        self.CHART_HEIGHT = int(self.HEIGHT * 0.6)  # 图表占60%高度
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.RESIZABLE)

    def _draw_chart(self):
        """
        绘制K线图
        """
        # 绘制K线图区域背景
        chart_rect = pygame.Rect(0, 0, self.WIDTH, self.CHART_HEIGHT)
        pygame.draw.rect(self.screen, (30, 30, 50), chart_rect)

        # 确定绘制范围
        start_idx = self.start_index
        end_idx = self.end_index

        # 检查有效性
        if end_idx <= start_idx:
            return

        # 获取价格范围
        prices_window = self.data.iloc[start_idx:end_idx]
        min_price = prices_window[['low', 'open', 'close']].min().min()
        max_price = prices_window[['high', 'open', 'close']].max().max()

        if min_price == max_price:
            return

        # 应用缩放因子
        price_range = max_price - min_price
        padding = price_range * 0.05 / self.chart_scale  # 应用缩放
        min_price -= padding
        max_price += padding
        price_range = max_price - min_price

        # 计算成交量区域高度 (占图表区域的1/4)
        volume_height = self.CHART_HEIGHT // 4
        volume_area_top = self.CHART_HEIGHT - volume_height

        # 缓存计算结果以提高性能
        width_ratio = self.WIDTH / (end_idx - start_idx)
        height_ratio = (self.CHART_HEIGHT - volume_height) / price_range

        # 绘制布林带
        for i in range(start_idx, min(end_idx - 1, len(self.data) - 1)):
            if i < len(self.data) - 1:
                x1 = int((i - start_idx) * width_ratio)
                x2 = int((i + 1 - start_idx) * width_ratio)

                # 上轨
                upper_band_curr = self.data.iloc[i]['upper_band']
                upper_band_next = self.data.iloc[i + 1]['upper_band']
                if not (np.isnan(upper_band_curr) or np.isnan(upper_band_next)):
                    y1 = volume_area_top - int((upper_band_curr - min_price) * height_ratio)
                    y2 = volume_area_top - int((upper_band_next - min_price) * height_ratio)
                    pygame.draw.line(self.screen, (100, 100, 200), (x1, y1), (x2, y2), 1)

                # 下轨
                lower_band_curr = self.data.iloc[i]['lower_band']
                lower_band_next = self.data.iloc[i + 1]['lower_band']
                if not (np.isnan(lower_band_curr) or np.isnan(lower_band_next)):
                    y1 = volume_area_top - int((lower_band_curr - min_price) * height_ratio)
                    y2 = volume_area_top - int((lower_band_next - min_price) * height_ratio)
                    pygame.draw.line(self.screen, (100, 100, 200), (x1, y1), (x2, y2), 1)

                # 中轨
                ma_curr = self.data.iloc[i]['ma']
                ma_next = self.data.iloc[i + 1]['ma']
                if not (np.isnan(ma_curr) or np.isnan(ma_next)):
                    y1 = volume_area_top - int((ma_curr - min_price) * height_ratio)
                    y2 = volume_area_top - int((ma_next - min_price) * height_ratio)
                    pygame.draw.line(self.screen, (150, 150, 220), (x1, y1), (x2, y2), 1)

        # 绘制K线
        bar_width = max(1, int(width_ratio) - 2)
        for i in range(start_idx, end_idx):
            x = int((i - start_idx + 0.5) * width_ratio)

            row = self.data.iloc[i]
            open_y = volume_area_top - int((row['open'] - min_price) * height_ratio)
            close_y = volume_area_top - int((row['close'] - min_price) * height_ratio)
            high_y = volume_area_top - int((row['high'] - min_price) * height_ratio)
            low_y = volume_area_top - int((row['low'] - min_price) * height_ratio)

            # 绘制最高最低价线
            pygame.draw.line(self.screen, (200, 200, 200), (x, high_y), (x, low_y), 1)

            # 绘制开盘收盘价矩形
            if row['close'] > row['open']:
                # 阳线（涨）
                color = (200, 100, 100)
                pygame.draw.rect(self.screen, color,
                                 (x - bar_width // 2, close_y, bar_width, open_y - close_y))
            elif row['close'] < row['open']:
                # 阴线（跌）
                color = (100, 200, 100)
                pygame.draw.rect(self.screen, color,
                                 (x - bar_width // 2, open_y, bar_width, close_y - open_y))
            else:
                # 十字星（开盘价等于收盘价）
                color = (255, 255, 255)  # 白色
                pygame.draw.line(self.screen, color,
                                 (x - bar_width // 2, close_y), (x + bar_width // 2, close_y), 1)

        # 绘制成交量
        max_volume = prices_window['volume'].max()
        if max_volume > 0:
            volume_height_scale = (volume_height - 10) / max_volume
            for i in range(start_idx, end_idx):
                x = int((i - start_idx + 0.5) * width_ratio)
                row = self.data.iloc[i]
                volume_height_px = int(row['volume'] * volume_height_scale)

                # 根据价格涨跌确定成交量颜色
                color = (200, 100, 100) if row['close'] >= row['open'] else (100, 200, 100)  # 绿色或红色

                pygame.draw.rect(self.screen, color,
                                 (x - bar_width // 2, self.CHART_HEIGHT - volume_height_px,
                                  bar_width, volume_height_px))

        # 绘制交易标记
        for trade in self.trade_history:
            if start_idx <= trade['index'] < end_idx:
                x = int((trade['index'] - start_idx + 0.5) * width_ratio)
                price_y = volume_area_top - int((trade['price'] - min_price) * height_ratio)

                if trade['action'] == 'buy_open':
                    # 买入开多标记 - 红向上箭头
                    pygame.draw.polygon(self.screen, (255, 0, 0), [
                        (x, price_y - 10),
                        (x - 10, price_y),
                        (x + 10, price_y)
                    ])
                elif trade['action'] == 'sell_open':
                    # 卖出开空标记 - 绿色向下箭头
                    pygame.draw.polygon(self.screen, (0, 255, 0), [
                        (x, price_y + 10),
                        (x - 10, price_y),
                        (x + 10, price_y)
                    ])
                elif trade['action'] == 'close_long':
                    pygame.draw.polygon(self.screen, (0, 255, 0), [
                        (x, price_y + 10),
                        (x - 10, price_y),
                        (x + 10, price_y)
                    ])
                elif trade['action'] == 'close_short':
                    pygame.draw.polygon(self.screen, (255, 0, 0), [
                        (x, price_y - 10),
                        (x - 10, price_y),
                        (x + 10, price_y)
                    ])

        # 在最右侧绘制带透明度的黄线表示当前K线位置
        if end_idx > start_idx:
            # 创建一个临时Surface用于绘制带透明度的线
            overlay = pygame.Surface((self.WIDTH, self.CHART_HEIGHT), pygame.SRCALPHA)
            # 计算最右侧K线的x坐标
            current_x = int((end_idx - 1 - start_idx + 0.5) * width_ratio)
            # 绘制黄色半透明竖线
            pygame.draw.line(overlay, (255, 255, 0, 128),
                             (current_x, 0), (current_x, self.CHART_HEIGHT), 2)
            # 将带透明度的线绘制到主屏幕
            self.screen.blit(overlay, (0, 0))

    def _draw_info_panel(self):
        """
        绘制信息面板
        """
        info_height = 120
        info_rect = pygame.Rect(0, self.CHART_HEIGHT, self.WIDTH, info_height)
        pygame.draw.rect(self.screen, (40, 40, 60), info_rect)

        # 如果有数据，显示最后一条数据的信息
        if len(self.data) > 0 and self.end_index > 0:
            last_row = self.data.iloc[self.current_kline_index]

            # 显示行情信息
            info_texts = [
                f"时间: {last_row['date'].strftime('%H:%M')}",
                f"开盘: {last_row['open']:.2f}",
                f"最高: {last_row['high']:.2f}",
                f"最低: {last_row['low']:.2f}",
                f"收盘: {last_row['close']:.2f}",
                f"成交量: {last_row['volume']:,.0f}"
            ]

            for i, text in enumerate(info_texts):
                text_surface = self.font.render(text, True, (220, 220, 220))
                self.screen.blit(text_surface, (30, self.CHART_HEIGHT +5 + i * 18))

        # 显示控制说明
        controls = [
            "↑/↓ : 缩放图表",
            "B : 买入(开多/平空)",
            "S : 卖出(开空/平多)",
            "ESC : 退出   提示:如键入无响应请切换输入法！！"

        ]
        # 计算控制说明的起始位置
        control_x = self.WIDTH - 600  # 从右侧开始
        for i, text in enumerate(controls):
            text_surface = self.small_font.render(text, True, (180, 180, 200))
            self.screen.blit(text_surface, (control_x, self.CHART_HEIGHT + 10 + i * 20))

    def _draw_trading_panel(self):
        """绘制交易信息面板"""
        panel_top = self.CHART_HEIGHT + 120
        panel_height = self.HEIGHT - panel_top
        panel_rect = pygame.Rect(0, panel_top, self.WIDTH, panel_height)
        pygame.draw.rect(self.screen, (50, 50, 70), panel_rect)

        # 显示账户信息
        account_info = [
            f"初始资金: {self.initial_capital:,.2f}",
            f"可用资金: {self.cash:,.2f}",
            f"持仓数量: {abs(self.position)}",
            f"持仓方向: {self.position_type}",
            f"持仓均价: {self.avg_cost:.2f}" if self.position != 0 else "持仓均价: 0.00",
            f"浮动盈亏: {self.floating_pnl:+.2f}",
            f"总资产: {self.total_value:,.2f}",
            f"当前步数: {self.current_step}/{self.max_steps}"
        ]

        for i, text in enumerate(account_info):
            color = (220, 220, 220)
            if "持仓方向" in text:
                if "多头" in text:
                    color = (0, 255, 0)
                elif "空头" in text:
                    color = (255, 0, 0)
            elif "浮动盈亏" in text:
                color = (0, 255, 0) if self.floating_pnl >= 0 else (255, 0, 0)
            elif "总资产" in text:
                color = (255, 255, 0)

            text_surface = self.font.render(text, True, color)
            self.screen.blit(text_surface, (20, panel_top + 10 + i * 25))

        # 显示当前可执行操作
        if self.position == 0:
            action_text = "当前可执行操作: 买入开多(B) 或 卖出开空(S)"
        elif self.position > 0:
            action_text = "当前可执行操作: 卖出平多(S)"
        else:
            action_text = "当前可执行操作: 买入平空(B)"

        action_surface = self.small_font.render(action_text, True, (200, 200, 200))
        self.screen.blit(action_surface, (300, panel_top + 10))

        # 显示最近交易记录
        recent_trades = self.trade_history[-5:]  # 显示最近5笔交易
        trade_text = self.small_font.render("最近交易:", True, (220, 220, 220))
        self.screen.blit(trade_text, (300, panel_top + 40))

        for i, trade in enumerate(recent_trades):
            action_map = {
                'buy_open': '买入开多',
                'sell_open': '卖出开空',
                'close_long': '卖出平多',
                'close_short': '买入平空'
            }
            action_text = action_map.get(trade['action'], trade['action'])
            trade_info = f"{trade['time'].strftime('%H:%M')} {action_text} {trade['volume']}手 @ {trade['price']:.2f}"
            trade_surface = self.small_font.render(trade_info, True, (200, 200, 200))
            self.screen.blit(trade_surface, (300, panel_top + 65 + i * 20))

    def buy_action(self):
        """买入操作"""
        # 检查是否达到最大步数
        if self.current_step >= self.max_steps:
            # 达到最大步数，强制平仓
            if self.position > 0:  # 多头持仓
                self._close_long(self.data.iloc[self.current_kline_index]['close'])
            elif self.position < 0:  # 空头持仓
                self._close_short(self.data.iloc[self.current_kline_index]['close'])
            return

        if self.current_kline_index >= len(self.data):
            return

        current_price = self.data.iloc[self.current_kline_index]['close']

        if self.position == 0:  # 无持仓，买入开多
            self._buy_open(current_price)
        elif self.position < 0:  # 有空头持仓，买入平空
            self._close_short(current_price)
        # 如果有多头持仓，不允许买入

    def sell_action(self):
        """卖出操作"""
        # 检查是否达到最大步数
        if self.current_step >= self.max_steps:
            # 达到最大步数，强制平仓
            if self.position > 0:  # 多头持仓
                self._close_long(self.data.iloc[self.current_kline_index]['close'])
            elif self.position < 0:  # 空头持仓
                self._close_short(self.data.iloc[self.current_kline_index]['close'])
            return

        if self.current_kline_index >= len(self.data):
            return

        current_price = self.data.iloc[self.current_kline_index]['close']

        if self.position == 0:  # 无持仓，卖出开空
            self._sell_open(current_price)
        elif self.position > 0:  # 有多头持仓，卖出平多
            self._close_long(current_price)
        # 如果有空头持仓，不允许卖出

    def _buy_open(self, price):
        """买入开多仓"""
        cost = self.trade_volume * price
        if self.cash >= cost:
            self.position = self.trade_volume
            self.avg_cost = price
            self.cash -= cost
            self.position_type = "多头"

            # 记录交易
            self.trade_history.append({
                'time': self.data.iloc[self.current_kline_index]['date'],
                'action': 'buy_open',
                'price': price,
                'volume': self.trade_volume,
                'index': self.current_kline_index
            })

            self._update_total_value()

    def _sell_open(self, price):
        """卖出开空仓"""
        # 开空仓需要保证金，这里简化处理，使用与开多相同的资金要求
        cost = self.trade_volume * price
        if self.cash >= cost:
            self.position = -self.trade_volume
            self.avg_cost = price
            self.cash -= cost  # 开空仓也需要冻结保证金
            self.position_type = "空头"

            # 记录交易
            self.trade_history.append({
                'time': self.data.iloc[self.current_kline_index]['date'],
                'action': 'sell_open',
                'price': price,
                'volume': self.trade_volume,
                'index': self.current_kline_index
            })

            self._update_total_value()

    def _close_long(self, price):
        """平多仓"""
        if self.position <= 0:
            return

        # 计算盈亏
        pnl = self.position * (price - self.avg_cost)
        self.cash += self.position * price

        # 记录交易
        self.trade_history.append({
            'time': self.data.iloc[self.current_kline_index]['date'],
            'action': 'close_long',
            'price': price,
            'volume': self.position,
            'pnl': pnl,
            'index': self.current_kline_index
        })

        # 重置持仓
        self.position = 0
        self.avg_cost = 0
        self.position_type = "空仓"

        self._update_total_value()

    def _close_short(self, price):
        """平空仓"""
        if self.position >= 0:
            return

        # 计算盈亏
        pnl = abs(self.position) * (self.avg_cost - price)
        self.cash += abs(self.position) * (2 * self.avg_cost - price)  # 返还保证金并结算盈亏

        # 记录交易
        self.trade_history.append({
            'time': self.data.iloc[self.current_kline_index]['date'],
            'action': 'close_short',
            'price': price,
            'volume': abs(self.position),
            'pnl': pnl,
            'index': self.current_kline_index
        })

        # 重置持仓
        self.position = 0
        self.avg_cost = 0
        self.position_type = "空仓"

        self._update_total_value()

    def _update_floating_pnl(self):
        """更新浮动盈亏"""
        if self.position == 0:
            self.floating_pnl = 0
        else:
            current_price = self.data.iloc[self.current_kline_index]['close']
            if self.position > 0:  # 多头持仓
                self.floating_pnl = self.position * (current_price - self.avg_cost)
            else:  # 空头持仓
                self.floating_pnl = abs(self.position) * (self.avg_cost - current_price)

        self._update_total_value()

    def _update_total_value(self):
        """更新总资产"""
        if self.position == 0:
            self.total_value = self.cash
        else:
            current_price = self.data.iloc[self.current_kline_index]['close']
            if self.position > 0:  # 多头持仓
                self.total_value = self.cash + self.position * current_price
            else:  # 空头持仓
                # 空头持仓的总价值计算：现金 + 持仓市值（负值） + 浮动盈亏
                self.total_value = self.cash + abs(self.position) * (2 * self.avg_cost - current_price)


def main():
    """主函数"""
    # 查找data目录下的CSV文件
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    if os.path.exists(data_dir):
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if csv_files:
            data_path = os.path.join(data_dir, csv_files[0])
            print(f"使用默认数据文件: {data_path}")
        else:
            print("在data目录中未找到CSV文件")
            return
    else:
        print("data目录不存在")
        return

    # 创建并运行K线查看器
    viewer = KLineViewer(data_path, max_steps=5000)
    viewer.run()


if __name__ == "__main__":
    main()