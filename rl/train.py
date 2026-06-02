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


import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class BananagramsFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        self.embedding = nn.Embedding(num_embeddings=27, embedding_dim=16)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        
        cnn_out_dim = 64 * 5 * 5 # 1600
        scalar_in_dim = 27 # 26 for hand + 1 for bag_count
        
        self.linear = nn.Sequential(
            nn.Linear(cnn_out_dim + scalar_in_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        grid = observations["grid"]
        if len(grid.shape) == 2:
            grid = grid.unsqueeze(0)
        if len(grid.shape) == 4 and grid.shape[1] == 1:
            grid = grid.squeeze(1)
        if len(grid.shape) == 4 and grid.shape[3] == 1:
            grid = grid.squeeze(3)
            
        grid = grid.long()
        embedded = self.embedding(grid)
        embedded = embedded.permute(0, 3, 1, 2)
        
        cnn_out = self.cnn(embedded)
        
        hand = observations["hand"].float()
        bag = observations["bag_count"].float()
        
        if len(hand.shape) == 1:
            hand = hand.unsqueeze(0)
            bag = bag.unsqueeze(0)
            
        combined = torch.cat([cnn_out, hand, bag], dim=1)
        return self.linear(combined)

def train(timesteps: int, save_dir: str, n_envs: int, overfit: bool = False):
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
        from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
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

    def make_eval_env():
        env = BananagramsEnv()
        env = ActionMasker(env, lambda e: e.action_masks())
        return env

    eval_env = DummyVecEnv([make_eval_env])

    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=40,  # ~2,000,000 total steps patience
        min_evals=60,                 # Grace period: ~3,000,000 total steps before stopping can trigger
        verbose=1
    )

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=max(1, 50_000 // actual_n_envs),
        callback_after_eval=stop_train_callback,
        best_model_save_path=os.path.join(save_dir, 'trained_models', 'best_model'),
        deterministic=True,
        n_eval_episodes=10,
        verbose=1
    )

    policy_kwargs = dict(
        features_extractor_class=BananagramsFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=512),
    )

    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=os.path.join(save_dir, "tb_logs"),
        n_steps=512,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        learning_rate=3e-4,
    )

    mode_str = "OVERFIT mode (1 env, fixed seed 42)" if overfit else f"{actual_n_envs} parallel envs"
    print(f"Training for {timesteps:,} timesteps across {mode_str}...")
    
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=eval_callback,
            progress_bar=False,
        )
    except (Exception, KeyboardInterrupt) as e:
        print(f"Training interrupted: {e}")

    best_model_path = os.path.join(save_dir, 'trained_models', 'best_model', 'best_model.zip')
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path} before saving final...")
        model = MaskablePPO.load(best_model_path, env=vec_env)

    final_model_path = os.path.join(save_dir, 'trained_models', "final_model")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}.zip")
    vec_env.close()

    try:
        from eval_model import graph_metrics
        monitor_csv = os.path.join(save_dir, "monitor.monitor.csv")
        plot_out = os.path.join(save_dir, "metrics", "learning_curve.png")
        os.makedirs(os.path.join(save_dir, "metrics"), exist_ok=True)
        if os.path.exists(monitor_csv):
            graph_metrics(monitor_csv, plot_out)
    except Exception as e:
        print(f"Could not automatically graph metrics: {e}")

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
