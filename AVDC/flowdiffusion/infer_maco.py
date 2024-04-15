from argparse import ArgumentParser
import json

from train_maco import TDWMacoTrainer
from utils import Visualizer

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--output", type=str, default="../results/tdw_maco_single/inference")
    parser.add_argument("--guidance_weight", type=float, default=5)
    args = parser.parse_args()
    setattr(args, "mode", "inference")
    setattr(args, "checkpoint_num", args.model)
    setattr(args, "result_dir", None)
    setattr(args, "inpainting", False)
    setattr(args, "single", True)
    setattr(args, "debug", False)
    setattr(args, "save_milestone", False)
    setattr(args, "no_superres", True)
    trainer = TDWMacoTrainer(args)
    with open(args.text, "r") as f:
        goal = json.load(f) # list of text goals
    vis = [Visualizer() for _ in range(len(goal))]
    trainer.inference(goal, args.image, args.output, True, vis)
    for i in range(len(goal)):
        vis[i].vis_attn_map(goal[i], f"{args.output}/attn_{i}.png")