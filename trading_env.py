"""交易环境模块，实现了Gymnasium环境接口用于强化学习训练"""

from typing import Literal
import pygame
import numpy as np
from gymnasium import Env, spaces
from interactive_trading_game import KLineViewer
import os


class TradingEnv(Env):
    """交易环境类，继承自Gymnasium的Env基类"""
    
    # 环境元数据，定义支持的渲染模式和帧率
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}
    
    def __init__(self, render_mode: Literal["human", "rgb_array"] = "human", max_steps=5000):
        """初始化交易环境
        
        Args:
            render_mode: 渲染模式，"human"表示实时渲染，"rgb_array"表示返回RGB数组
            max_steps: 最大步数，达到该步数后强制平仓并结束
        """
        # 调用父类构造函数
        super().__init__()
        
        # 存储渲染模式
        self.render_mode = render_mode
        
        # 最大步数
        self.max_steps = max_steps
        
        # 查找data目录下的CSV文件
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        if os.path.exists(data_dir):
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            if csv_files:
                data_path = os.path.join(data_dir, csv_files[0])
            else:
                raise FileNotFoundError("在data目录中未找到CSV文件")
        else:
            raise FileNotFoundError("data目录不存在")
        
        # 创建交易游戏实例
        self.game = KLineViewer(data_path)

        # 定义动作空间为离散空间，大小为3（0: 无操作, 1: 买入, 2: 卖出）
        self.action_space = spaces.Discrete(3)
        
        # 定义简化的观测空间为连续空间
        # 包括：当前价格信息(5个值) + 技术指标(3个值) + 趋势指标(3个值) + 账户信息(5个值) = 16个值
        # 价格信息：开盘价、最高价、最低价、收盘价、成交量
        # 技术指标：MA、上轨、下轨
        # 趋势指标：价格变化率、MA差值、布林带宽度
        # 账户信息：可用资金、持仓数量、持仓均价、浮动盈亏、总资产
        self.observation_space = spaces.Box(
            low=-np.inf,       # 观测值下界
            high=np.inf,       # 观测值上界
            shape=(16,),       # 扩展后的观测空间形状
            dtype=np.float32,  # 数据类型
        )
        
        # 记录上一次交易历史的长度
        self.last_trade_count = 0
        
        # 记录初始资金
        self.initial_cash = 0
        
        # 记录上一步的总价值
        self.last_total_value = 0
        
        # 记录步数
        self.current_step = 0
        
        # 记录交易次数
        self.total_trades = 0
        
        # 重置环境到初始状态
        self.reset()

    def reset(self, *, seed=None, options=None):
        """重置环境到初始状态

        Args:
            seed: 随机种子
            options: 其他选项参数

        Returns:
            tuple: (observation, info) 观测值和附加信息
        """
        # 调用父类重置方法并传递随机种子
        super().reset(seed=seed)

        # 重新随机截取行情数据
        self.game._random_slice()

        # 重置交易系统
        self.game._init_trading_system()

        # 重置交易记录
        self.last_trade_count = 0

        # 记录初始资金
        self.initial_cash = self.game.initial_capital

        # 记录初始总价值
        self.last_total_value = self.game.total_value

        # 重置步数
        self.current_step = 0

        # 重置交易次数
        self.total_trades = 0

        # 获取当前观测值
        observation = self._get_obs()

        # 获取附加信息
        info = self._get_info()

        # 返回观测值和信息
        return observation, info

    def step(self, action):
        """执行一个动作并推进环境状态

        Args:
            action: 要执行的动作 (0: 无操作, 1: 买入, 2: 卖出)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
                   观测值、奖励、是否终止、是否截断、附加信息
        """
        # 确保动作在动作空间内
        assert self.action_space.contains(action)

        # 增加步数
        self.current_step += 1

        # 记录执行动作前的交易历史数量
        self.last_trade_count = len(self.game.trade_history)

        # 记录执行动作前的总价值
        self.last_total_value = self.game.total_value

        # 获取动作前的状态
        prev_position = self.game.position

        # 执行动作，直接调用游戏方法
        if action == 1:  # 买入
            self.game.buy_action()
        elif action == 2:  # 卖出
            self.game.sell_action()
        # action == 0 为无操作，不需要执行任何函数

        # 自动向右滚动一根K线
        self.game._scroll(1)

        # 更新浮动盈亏
        self.game._update_floating_pnl()

        # 更新交易次数
        if len(self.game.trade_history) > self.last_trade_count:
            self.total_trades += 1

        # 改进奖励机制
        # 1. 基于总资产变化的奖励
        total_value_change = self.game.total_value - self.last_total_value
        # 归一化奖励，基于初始资金的比例
        reward = total_value_change / self.initial_cash * 10000

        # 2. 添加交易激励项，鼓励有效交易
        if len(self.game.trade_history) > self.last_trade_count:
            reward += 0.1  # 对每次成功交易给予小奖励

        # 3. 添加最终奖励
        # 判断是否达到最大步数
        max_steps_reached = self.current_step >= self.max_steps

        # 如果达到最大步数，强制平仓并给与最终奖励
        if max_steps_reached:
            # 强制平仓
            if self.game.position > 0:  # 多头持仓
                self.game.sell_action()  # 卖出平多
            elif self.game.position < 0:  # 空头持仓
                self.game.buy_action()  # 买入平空

            # 给予最终奖励
            final_reward = (self.game.total_value - self.initial_cash) / self.initial_cash * 10000
            reward += final_reward

        # 对奖励进行裁剪，防止过大或过小
        reward = np.clip(reward, -100.0, 100.0)

        # 获取新的观测值
        observation = self._get_obs()

        # 判断是否终止（数据播放完毕或达到最大步数）
        terminated = (self.game.current_kline_index >= len(self.game.data) - 2) or max_steps_reached

        # 是否因时间限制而截断（此处始终为False）
        truncated = False

        # 获取附加信息
        info = self._get_info()

        # 返回结果元组
        return observation.astype(np.float32), reward, terminated, truncated, info

    def render(self):
        """渲染环境

        Returns:
            None or numpy.ndarray: 根据渲染模式返回相应内容
        """
        # 人类可读模式：直接绘制游戏画面
        if self.render_mode == "human":
            self.game.render()
            return None

        # RGB数组模式：返回图像数组用于录像等用途
        if self.render_mode == "rgb_array":
            # 保存原始表面和屏幕对象
            original_surface = self.game.screen

            # 创建临时表面用于渲染
            temp_surface = pygame.Surface((self.game.WIDTH, self.game.HEIGHT))
            self.game.screen = temp_surface

            # 绘制游戏画面
            self.game.render()

            # 将表面转换为numpy数组
            array = pygame.surfarray.array3d(temp_surface).transpose(1, 0, 2)

            # 恢复原始表面和屏幕对象
            self.game.screen = original_surface

            # 返回图像数组
            return array

    def _get_info(self):
        """获取环境附加信息

        Returns:
            dict: 包含环境状态的信息
        """
        # 计算总收益率
        total_return = (self.game.total_value - self.initial_cash) / self.initial_cash

        return {
            "total_value": self.game.total_value,
            "position": self.game.position,
            "cash": self.game.cash,
            "current_index": self.game.current_kline_index,
            "total_return": total_return,
            "current_step": self.current_step,
            "total_trades": self.total_trades
        }

    def _get_obs(self):
        """获取当前观测值，扩展版本包含更多信息

        Returns:
            numpy.ndarray: 扩展后的观测值数组
        """
        # 确保索引有效
        idx = max(0, min(self.game.current_kline_index, len(self.game.data) - 1))

        # 获取当前K线数据
        current_row = self.game.data.iloc[idx]

        # 获取前一根K线数据用于计算变化率
        if idx > 0:
            prev_row = self.game.data.iloc[idx-1]
        else:
            prev_row = current_row

        # 构造价格信息向量：开盘价、最高价、最低价、收盘价、成交量
        price_info = np.array([
            current_row['open'],              # 开盘价
            current_row['high'],              # 最高价
            current_row['low'],               # 最低价
            current_row['close'],             # 收盘价
            current_row['volume'],            # 成交量
        ], dtype=np.float32)

        # 构造技术指标向量：MA、上轨、下轨
        tech_info = np.array([
            current_row['ma'] if not np.isnan(current_row['ma']) else 0,              # MA
            current_row['upper_band'] if not np.isnan(current_row['upper_band']) else 0,  # 上轨
            current_row['lower_band'] if not np.isnan(current_row['lower_band']) else 0,  # 下轨
        ], dtype=np.float32)

        # 构造趋势指标向量：价格变化率、MA差值、布林带宽度
        price_change_rate = (current_row['close'] - prev_row['close']) / prev_row['close'] if prev_row['close'] != 0 else 0
        ma_diff = current_row['close'] - current_row['ma'] if not np.isnan(current_row['ma']) else 0
        bollinger_width = (current_row['upper_band'] - current_row['lower_band']) if not (np.isnan(current_row['upper_band']) or np.isnan(current_row['lower_band'])) else 0

        trend_info = np.array([
            price_change_rate,      # 价格变化率
            ma_diff,                # 收盘价与MA的差值
            bollinger_width,        # 布林带宽度
        ], dtype=np.float32)

        # 构造账户信息向量：可用资金、持仓数量、持仓均价、浮动盈亏、总资产
        account_info = np.array([
            self.game.cash,                   # 可用资金
            self.game.position,               # 持仓数量
            self.game.avg_cost,               # 持仓均价
            self.game.floating_pnl,           # 浮动盈亏
            self.game.total_value,            # 总资产
        ], dtype=np.float32)

        # 将所有信息合并为单一观测向量
        return np.concatenate([price_info, tech_info, trend_info, account_info])