import argparse
from pathlib import Path
import re

from trading_env import TradingEnv
from stable_baselines3 import PPO
import pygame


def eval_agent(model_path=None, max_steps=5000):
    """评估智能体表现的函数
    
    Args:
        model_path: 模型文件路径
        max_steps: 最大步数
    """
    print(f"正在使用模型: {model_path}")
    
    # 创建环境
    env = TradingEnv(render_mode="human", max_steps=max_steps)
    # 从指定路径加载预训练的PPO模型
    model = PPO.load(model_path, env=env)
    obs, _ = env.reset()

    # 控制渲染速度
    clock = pygame.time.Clock()
    
    # 进行评估直到环境结束
    step = 0
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        # 使用模型预测当前状态下应采取的动作
        action, _ = model.predict(obs, deterministic=True)
        # 执行动作并获取环境反馈
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 渲染环境画面
        env.render()
        # 处理Pygame事件以防止界面冻结，并控制帧率
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    terminated = True
                    break
        # 控制帧率为30 FPS，使演示效果更好
        clock.tick(30)
                
        step += 1
        # 每100步输出一次信息
        if step % 100 == 0:
            print(f"Step: {step}, Total Value: {info['total_value']:.2f}, Position: {info['position']}")
            
    # 输出最终结果
    print(f"评估结束! 总步数: {step}")
    print(f"最终资产: {info['total_value']:.2f}")
    print(f"初始资金: {env.initial_cash:.2f}")
    print(f"收益率: {(info['total_value'] - env.initial_cash) / env.initial_cash * 100:.2f}%")
    
    # 关闭环境
    env.close()


def _main():
    """主函数"""
    parser = argparse.ArgumentParser()
    # 添加模型文件路径参数
    parser.add_argument("--model", type=str)
    # 添加最大步数参数
    parser.add_argument("--max-steps", type=int, default=5000)
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        # 如果未指定模型路径，则自动查找最新的模型文件
        log_dir = Path("tmp")
        model_files = list(log_dir.glob("trading_rl_model*.zip")) if log_dir.exists() else []

        # 按照步数排序，选择最新的模型文件
        def extract_step_count(filename):
            match = re.search(r'(\d+)_steps', filename.stem)
            return int(match.group(1)) if match else 0
        
        if model_files:
            model_files.sort(key=extract_step_count)
            model_path = str(model_files[-1])
            print(f"选择最新的模型: {model_path}")
        else:
            print("在 tmp/ 目录中未找到任何模型文件")
            return

    # 调用评估函数
    eval_agent(model_path=model_path, max_steps=args.max_steps)


# 程序入口点
if __name__ == "__main__":
    _main()