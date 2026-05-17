"""
Training script — Bananagrams RL agent using MaskablePPO.

Requirements:
    pip install gymnasium stable-baselines3 sb3-contrib

Usage:
    python train.py                        # train with defaults
    python train.py --timesteps 2000000    # longer run
    python train.py --load runs/best_model # resume / evaluate
"""

import argparse
import os

import numpy as np
from gymnasium.wrappers import FlattenObservation

from env import BananagramsEnv


def make_env(render_mode=None):
    """Factory used by SubprocVecEnv."""
    def _init():
        return BananagramsEnv(render_mode=render_mode)
    return _init


def train(timesteps: int, save_dir: str, n_envs: int):
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.utils import get_action_masks
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    except ImportError:
        raise SystemExit(
            "Missing dependencies. Run:\n"
            "  pip install stable-baselines3 sb3-contrib"
        )

    os.makedirs(save_dir, exist_ok=True)

    def _wrapped_env():
        env = BananagramsEnv()
        # ActionMasker tells SB3 how to retrieve the mask each step
        env = ActionMasker(env, lambda e: e.action_masks())
        return env

    vec_env = SubprocVecEnv([_wrapped_env] * n_envs)
    vec_env = VecMonitor(vec_env, filename=os.path.join(save_dir, "monitor"))

    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=os.path.join(save_dir, "tb_logs"),
        # Hyperparameters — reasonable starting point, tune as needed
        n_steps=512,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,      # encourage exploration
        learning_rate=3e-4,
    )

    print(f"Training for {timesteps:,} timesteps across {n_envs} parallel envs...")
    model.learn(
        total_timesteps=timesteps,
        progress_bar=True,
    )

    model.save(os.path.join(save_dir, "final_model"))
    print(f"Model saved to {save_dir}/final_model.zip")
    vec_env.close()
    return model


def evaluate(model_path: str, n_episodes: int = 10):
    """Run a saved model and print episode stats."""
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
    except ImportError:
        raise SystemExit("Run: pip install stable-baselines3 sb3-contrib")

    env = BananagramsEnv(render_mode="human")
    env = ActionMasker(env, lambda e: e.action_masks())
    model = MaskablePPO.load(model_path, env=env)

    wins, total_rewards, total_steps = 0, [], []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done, ep_reward, steps = False, 0.0, 0
        while not done:
            action, _ = model.predict(obs, action_masks=env.action_masks())
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated
        if info.get("won"):
            wins += 1
        total_rewards.append(ep_reward)
        total_steps.append(steps)
        print(f"  Episode {ep+1:3d}: reward={ep_reward:7.2f}  steps={steps:5d}  won={info.get('won')}")

    print(f"\nWin rate: {wins}/{n_episodes}  |  "
          f"Avg reward: {np.mean(total_rewards):.2f}  |  "
          f"Avg steps: {np.mean(total_steps):.0f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/evaluate Bananagrams RL agent")
    parser.add_argument("--timesteps", type=int,   default=500_000,
                        help="Total training timesteps (default: 500000)")
    parser.add_argument("--envs",      type=int,   default=4,
                        help="Number of parallel environments (default: 4)")
    parser.add_argument("--save-dir",  type=str,   default="runs",
                        help="Directory to save model + logs (default: runs/)")
    parser.add_argument("--load",      type=str,   default=None,
                        help="Path to a saved model to evaluate instead of training")
    parser.add_argument("--eval-eps",  type=int,   default=10,
                        help="Episodes to run during evaluation (default: 10)")
    args = parser.parse_args()

    if args.load:
        evaluate(args.load, n_episodes=args.eval_eps)
    else:
        train(timesteps=args.timesteps, save_dir=args.save_dir, n_envs=args.envs)
