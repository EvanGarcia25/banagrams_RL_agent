"""
Training script - Bananagrams RL agent using MaskablePPO.

Requirements:
    pip install gymnasium stable-baselines3 sb3-contrib

Usage:
    python -m rl.train                        # train with defaults
    python -m rl.train --timesteps 2000000    # longer run
    python -m rl.train --load runs/best_model # resume / evaluate
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .env import BananagramsEnv


def make_env(render_mode=None):
    """Factory used by SubprocVecEnv."""
    def _init():
        return BananagramsEnv(render_mode=render_mode)
    return _init


def train(timesteps: int, save_dir: str, n_envs: int, overfit: bool = False):
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    except ImportError:
        raise SystemExit(
            "Missing dependencies. Run:\n"
            "  pip install stable-baselines3 sb3-contrib"
        )

    os.makedirs(save_dir, exist_ok=True)

    def make_env(rank):
        def _init():
            import gymnasium as gym
            class FixedSeedWrapper(gym.Wrapper):
                def __init__(self, env, seed):
                    super().__init__(env)
                    self._fixed_seed = seed
                def reset(self, **kwargs):
                    kwargs["seed"] = self._fixed_seed
                    return self.env.reset(**kwargs)
                def action_masks(self):
                    return self.env.action_masks()

            env = BananagramsEnv()
            if overfit:
                env = FixedSeedWrapper(env, 42)
            env = ActionMasker(env, lambda e: e.action_masks())
            return env
        return _init

    actual_n_envs = 1 if overfit else n_envs
    vec_env = SubprocVecEnv([make_env(i) for i in range(actual_n_envs)])
    vec_env = VecMonitor(vec_env, filename=os.path.join(save_dir, "monitor"))

    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=os.path.join(save_dir, "tb_logs"),
        n_steps=512,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        learning_rate=3e-4,
    )

    mode_str = "OVERFIT mode (1 env, fixed seed 42)" if overfit else f"{actual_n_envs} parallel envs"
    print(f"Training for {timesteps:,} timesteps across {mode_str}...")
    model.learn(
        total_timesteps=timesteps,
        progress_bar=False,
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate Bananagrams RL agent")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Total training timesteps (default: 500000)")
    parser.add_argument("--envs", type=int, default=4,
                        help="Number of parallel environments (default: 4)")
    parser.add_argument("--save-dir", type=str, default="runs",
                        help="Directory to save model + logs (default: runs/)")
    parser.add_argument("--load", type=str, default=None,
                        help="Path to a saved model to evaluate instead of training")
    parser.add_argument("--eval-eps", type=int, default=10,
                        help="Episodes to run during evaluation (default: 10)")
    parser.add_argument("--overfit", action="store_true",
                        help="Run an overfit test using a fixed seed (sets default timesteps to 100,000)")
    args = parser.parse_args(argv)

    if args.load:
        evaluate(args.load, n_episodes=args.eval_eps)
    else:
        timesteps = 100_000 if args.overfit and args.timesteps == 500_000 else args.timesteps
        train(timesteps=timesteps, save_dir=args.save_dir, n_envs=args.envs, overfit=args.overfit)


if __name__ == "__main__":
    main()
