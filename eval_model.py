import argparse
from pathlib import Path

from rl.env import BananagramsEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from viewer.episode_log import build_episode_record, make_run_metadata, append_event, finalize_episode_record, save_episode

import gymnasium as gym

class FixedSeedWrapper(gym.Wrapper):
    def __init__(self, env, seed):
        super().__init__(env)
        self._fixed_seed = seed
        
    def reset(self, **kwargs):
        if self._fixed_seed is not None:
            kwargs["seed"] = self._fixed_seed
        return self.env.reset(**kwargs)
        
    def action_masks(self):
        return self.env.action_masks()

def run_evaluation(model_path: str, seed: int | None, output_path: str, max_steps: int = 2000):
    # Setup environment
    env = BananagramsEnv(max_steps=max_steps)
    
    # Apply fixed seed logic if requested
    env = FixedSeedWrapper(env, seed)
    env = ActionMasker(env, lambda e: e.action_masks())
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = MaskablePPO.load(model_path, env=env)
    
    obs, _ = env.reset()
    
    run_meta = make_run_metadata(
        source="model_eval",
        seed=seed,
        model_name=Path(model_path).stem,
    )
    
    # We need initial state from the underlying game
    game = env.unwrapped.game
    record = build_episode_record(run_meta, game.get_state())
    
    print(f"Running evaluation episode...")
    done = False
    step_idx = 0
    ep_reward = 0.0
    
    while not done:
        action, _ = model.predict(obs, action_masks=env.action_masks())
        
        action_val = int(action)
        action_tuple = env.unwrapped.decode_action(action_val)
        
        # Manually format raw command based on tuple
        if action_tuple[0] == "place":
            raw_cmd = f"place {action_tuple[1]} {action_tuple[2]} {action_tuple[3]}"
            cmd_dict = {"kind": "place", "letter": action_tuple[1], "row": action_tuple[2], "col": action_tuple[3]}
        elif action_tuple[0] == "remove":
            raw_cmd = f"remove {action_tuple[1]} {action_tuple[2]}"
            cmd_dict = {"kind": "remove", "row": action_tuple[1], "col": action_tuple[2]}
        else:
            raw_cmd = f"dump {action_tuple[1]}"
            cmd_dict = {"kind": "dump", "letter": action_tuple[1]}
        
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        
        append_event(
            record,
            step_index=step_idx,
            raw_command=raw_cmd,
            command=cmd_dict,
            result={"success": info.get("success", False), "message": info.get("message", ""), "state": game.get_state()}
        )
        
        step_idx += 1
        done = terminated or truncated
        
    finalize_episode_record(record)
    save_path = save_episode(record, output_path)
    
    print(f"Episode complete. Won: {info.get('won', False)}, Steps: {step_idx}, Reward: {ep_reward:.2f}")
    print(f"Saved evaluation episode to {save_path}")

def graph_metrics(monitor_csv: str, plot_out: str):
    import pandas as pd
    import matplotlib.pyplot as plt
    print(f"Reading training metrics from {monitor_csv}...")
    try:
        df = pd.read_csv(monitor_csv, skiprows=1)
        plt.figure(figsize=(10, 5))
        model_name = Path(plot_out).stem
        ax = df["r"].rolling(10).mean().plot(
            title=f"Learning Curve: {model_name} (Rolling Mean 10)", 
            xlabel="Training Episode", 
            ylabel="Average Episode Reward"
        )
        ax.set_title(f"Learning Curve: {model_name}\n(Rolling Mean 10)", pad=10)
        plt.tight_layout()
        plt.savefig(plot_out)
        print(f"Saved training learning curve to {plot_out}")
        
        first_10 = df["r"].head(10).mean()
        last_10 = df["r"].tail(10).mean()
        print(f"Average reward in first 10 episodes: {first_10:.2f}")
        print(f"Average reward in last 10 episodes: {last_10:.2f}")
        
        if last_10 > first_10 + 20: # Arbitrary threshold to indicate some learning
            print("Conclusion: The model showed signs of learning over the course of training.")
        else:
            print("Conclusion: The model failed to meaningfully improve its score over the training period.")
    except Exception as e:
        print(f"Failed to graph metrics: {e}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained MaskablePPO model and save an episode log.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model (.zip)")
    parser.add_argument("--output", type=str, default="viewer/episodes/eval_episode.json", help="Path to save the JSON episode log")
    parser.add_argument("--seed", type=int, default=None, help="Fixed seed for the environment (optional)")
    parser.add_argument("--max-steps", type=int, default=2000, help="Maximum steps for the episode")
    parser.add_argument("--monitor-csv", type=str, default=None, help="Path to the training monitor.csv to graph metrics")
    parser.add_argument("--plot-out", type=str, default="runs/metrics/learning_curve.png", help="Path to save the metrics plot")
    args = parser.parse_args()
    
    run_evaluation(args.model, args.seed, args.output, args.max_steps)
    
    if args.monitor_csv:
        graph_metrics(args.monitor_csv, args.plot_out)

if __name__ == "__main__":
    main()
