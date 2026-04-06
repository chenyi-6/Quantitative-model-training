# benchmark_gpu_cpu.py
import time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import TradingEnv  # 你的交易环境


def benchmark_device(device_name):
    """测试指定设备的训练速度"""
    env = DummyVecEnv([lambda: TradingEnv(render_mode="rgb_array", max_steps=5000)])

    start_time = time.time()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,  # 不输出详细信息
        device=device_name,
        batch_size=64,
        n_steps=256,  # 减少步数以快速测试
    )

    # 训练一小段时间
    model.learn(total_timesteps=10000)

    elapsed_time = time.time() - start_time
    print(f"{device_name.upper()} 训练时间: {elapsed_time:.2f}秒")

    return elapsed_time


# 测试
print("=== 设备性能对比测试 ===")
cpu_time = benchmark_device("cpu")
gpu_time = benchmark_device("cuda")

if cpu_time < gpu_time:
    print(f"✅ CPU 比 GPU 快 {(gpu_time / cpu_time - 1) * 100:.1f}%")
else:
    print(f"✅ GPU 比 CPU 快 {(cpu_time / gpu_time - 1) * 100:.1f}%")