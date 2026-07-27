#!/usr/bin/env python3
"""Evaluate a saved SAC actor and record one off-screen episode as a GIF."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import gym
import numpy as np
from PIL import Image
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src" / "agent")]

# Importing the package registers the project's custom Gym environments.
import highway_env  # noqa: E402,F401
import core  # noqa: E402,F401  # Required by torch.load for this checkpoint format.


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "data/SAC_SACD-TDn-MPC-10th/SAC_SACD-TDn-MPC-10th_s3/pyt_save/model.pt",
        help="Path to the saved PyTorch actor.",
    )
    parser.add_argument("--env", default="merge_game_env-v0")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_result/SAC_SACD-TDn-MPC-10th_s3",
    )
    parser.add_argument("--gif-name", default="episode_001.gif")
    parser.add_argument("--gif-fps", type=float, default=5.0)
    parser.add_argument(
        "--no-safe-protect",
        action="store_true",
        help="Disable the MPC action shield used while training this checkpoint.",
    )
    return parser.parse_args()


def select_action(agent, observation):
    """Match the project's SAC evaluator: sample from the actor distribution."""
    with torch.no_grad():
        action, _, _, _ = agent.step(torch.from_numpy(observation).float())
    return int(action.item())


def save_gif(frames, path, fps):
    if not frames:
        raise RuntimeError("No frames were rendered; GIF was not created.")
    duration_ms = round(1000 / fps)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def main():
    args = parse_args()
    if args.eval_episodes < 1:
        raise ValueError("--eval-episodes must be at least 1")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be at least 1")
    if args.gif_fps <= 0:
        raise ValueError("--gif-fps must be positive")
    if not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    agent = torch.load(args.model_path, map_location="cpu")
    agent.eval()

    env = gym.make(args.env).unwrapped
    env.action_space.seed(args.seed)
    env.configure(
        {
            "simulation_frequency": 10,
            "policy_frequency": 2,
            "screen_width": 1200,
            "screen_height": 360,
            "scaling": 10,
            "mpc_control": True,
            "safe_check": not args.no_safe_protect,
            "offscreen_rendering": True,
            "real_time_rendering": False,
            "show_mpc_trajectory": True,
            "show_other_vehicles_predict": True,
        }
    )

    results = []
    frames = []
    use_safe_protect = not args.no_safe_protect
    try:
        for episode in range(args.eval_episodes):
            observation = env.reset()
            episode_cost = 0.0
            episode_return = 0.0
            crashed = False
            success = False
            # The MPC trajectory does not exist until the first action is applied,
            # so capture the first frame immediately after that action.
            episode_frames = []

            for step in range(1, args.max_steps + 1):
                flat_observation = observation[1:].flatten()
                action = select_action(agent, flat_observation)
                if use_safe_protect:
                    # Training used the shield's lightweight prediction mode.
                    action = env.check_action(action, check_type="simple")

                observation, reward, done, info = env.step(action)
                episode_return += float(reward)
                episode_cost += float(info["cost"])
                crashed = crashed or bool(info["crashed"])
                success = success or bool(info["success"])
                if episode == 0:
                    episode_frames.append(env.render(mode="rgb_array"))
                if done:
                    break

            if episode == 0:
                frames = episode_frames
            result = {
                "episode": episode + 1,
                "return": episode_return,
                "cost": episode_cost,
                "length": step,
                "crashed": crashed,
                "success": success,
            }
            results.append(result)
            print(
                "Episode {episode}: return={return:.3f}, cost={cost:.3f}, "
                "length={length}, crashed={crashed}, success={success}".format(**result)
            )
    finally:
        env.close()

    gif_path = args.output_dir / args.gif_name
    save_gif(frames, gif_path, args.gif_fps)
    summary = {
        "model_path": str(args.model_path),
        "environment": args.env,
        "seed": args.seed,
        "safe_protect": use_safe_protect,
        "eval_episodes": args.eval_episodes,
        "crash_ratio": sum(result["crashed"] for result in results) / len(results),
        "success_ratio": sum(result["success"] for result in results) / len(results),
        "average_return": float(np.mean([result["return"] for result in results])),
        "average_cost": float(np.mean([result["cost"] for result in results])),
        "average_length": float(np.mean([result["length"] for result in results])),
        "gif": str(gif_path),
        "gif_frames": len(frames),
        "episodes": results,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Saved GIF:", gif_path)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
