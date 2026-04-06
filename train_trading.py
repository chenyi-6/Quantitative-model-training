import argparse
import os
from pathlib import Path
import re

from trading_env import TradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env


def _main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加并行环境数量参数，默认值为1
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments",
    )
    # 添加总时间步数参数，默认值为100万
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=2_000_000,
        help="Number of timesteps to train for",
    )
    # 添加保存频率参数，默认每10000步保存一次检查点
    parser.add_argument(
        "--save-freq",
        type=int,
        default=100000,
        help="Frequency (in steps) to save checkpoints",
    )
    # 添加最大K线步数参数
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Maximum number of K-lines per episode",
    )
    # 解析所有命令行参数
    args = parser.parse_args()

    # =============================
    # 训练模式设置
    # "new" 表示从头开始训练
    # "resume" 表示从最新的检查点继续训练
    training_mode = "new"  # 修改这里来选择训练模式
    # =============================

    # 创建向量化环境，同时运行多个环境实例以加速训练
    vec_env = make_vec_env(
        TradingEnv,  # 环境类
        n_envs=args.n_envs,  # 并行环境数量
        env_kwargs={"render_mode": "rgb_array", "max_steps": args.max_steps},  # 环境初始化参数
        monitor_dir="tmp/",  # 监控数据保存目录
    )
    # 注意：make_vec_env 已经包装了 Monitor，不需要再手动添加 VecMonitor
    
    # 设置日志目录路径
    log_dir = "tmp/"
    # 创建日志目录，如果已存在则不报错
    os.makedirs(log_dir, exist_ok=True)

    # 根据模式决定是创建新模型还是加载已有模型
    if training_mode == "resume":
        # 查找最新的模型文件
        log_dir_path = Path("tmp")
        model_files = list(log_dir_path.glob("*.zip")) if log_dir_path.exists() else []
        if not model_files:
            print("No checkpoint found, starting new training")
            model_path = None
        else:
            # 按照步数排序，选择最新的模型文件
            def extract_step_count(filename):
                match = re.search(r'(\d+)_steps', filename.stem)
                return int(match.group(1)) if match else 0
            
            model_files.sort(key=extract_step_count)
            model_path = str(model_files[-1])
            print(f"Resuming training from {model_path}")
    else:
        model_path = None

    # 初始化PPO模型，如果指定了模型路径则加载，否则创建新模型
    if model_path:
        model = PPO.load(model_path, env=vec_env, device="cpu")
        # 继续训练需要将环境设置到模型中
        model.set_env(vec_env)
        reset_num_timesteps = False  # 继续训练时不重置时间步数
    else:
        model = PPO(
            "MlpPolicy",  # 使用多层感知机策略网络
            vec_env,  # 向量化环境
            verbose=1,  # 输出详细信息级别
            device="cpu",  # 在CPU上运行
            batch_size=64,  # 批处理大小
            # 调整参数以提高训练稳定性
            clip_range=0.2,              # 增加clip range以允许更大策略更新
            n_steps=512,                 # 增加步数以提高样本效率
            n_epochs=10,                 # 增加训练轮数
            learning_rate=3e-4,          # 使用更标准的学习率
            gamma=0.99,                  # discount factor
            gae_lambda=0.95,             # factor for trade-off of bias vs variance for Generalized Advantage Estimator
            ent_coef=0.05,               # 增加熵系数以鼓励探索
            vf_coef=0.5,                 # 标准价值函数系数
            max_grad_norm=0.5,           # 适度的梯度裁剪
            target_kl=0.03,              # 增加目标KL散度以允许更大更新
            clip_range_vf=None,          # 不裁剪价值函数
        )
        reset_num_timesteps = True  # 新训练重置时间步数

    # 设置TensorBoard日志名称
    tb_log_name = "ppo_trading"
    # 如果并行环境数大于0，则在日志名称中包含环境数信息
    if args.n_envs > 0:
        tb_log_name += f"_nenv{args.n_envs}"

    # 创建检查点回调对象，用于定期保存模型
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,  # 保存频率（步数）
        save_path="./tmp/",  # 保存路径
        name_prefix="trading_rl_model",  # 文件名前缀
        save_replay_buffer=True,  # 是否保存重放缓冲区
        save_vecnormalize=True,  # 是否保存向量归一化统计信息
    )

    # 开始模型训练
    model.learn(
        total_timesteps=args.total_timesteps,  # 总训练时间步数
        callback=checkpoint_callback,  # 回调函数，在每个训练步骤后调用
        tb_log_name=tb_log_name,  # TensorBoard日志名称
        reset_num_timesteps=reset_num_timesteps,  # 根据模式决定是否重置时间步数
    )


# 当脚本作为主程序运行时执行_main函数
if __name__ == "__main__":
    _main()