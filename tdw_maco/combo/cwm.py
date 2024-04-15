from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import os
import torch
from PIL import Image
from torchvision import transforms, utils
from einops import repeat
import requests
import json
import numpy as np

from AVDC.flowdiffusion.goal_diffusion import GoalGaussianDiffusion
from AVDC.flowdiffusion.unet import UnetMaco, UnetSuperRes, UnetTDWMacoInpainting, UNetModel
from AVDC.flowdiffusion.utils import Cache, Visualizer


class CWM:
    SERVER_ADDR = "http://localhost:8080"

    def __init__(self,
                serve=False,
                device="cuda:0",
                target_size = (128, 128),
                sample_per_seq=8,
                guidance_weight=5,
                model_id='',
                inpainting_model_id='',
                superres_model='',
                cache_root=''):
        self.serve = False
        self.device = device
        self.target_size = target_size
        self.superres_size = (336, 336)
        self.sample_per_seq = sample_per_seq
        self.guidance_weight = guidance_weight
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])
        self.inpainting_transform = transforms.Compose([
            transforms.Resize(self.superres_size),
            transforms.ToTensor(),
        ])
        if self.serve:
            return
        self.cache = Cache(root=cache_root)
        self.model = None
        self.superres_model = None
        self.inpainting_model = None
        self.init_vdm(model_id)
        self.init_superres(superres_model)
        self.init_inpainting_vdm(inpainting_model_id)

    def init_inpainting_vdm(self, model_id):
        print("Loading VDM topdown inpainting model...")
        unet = UnetTDWMacoInpainting(embed_dim=4096)

        diffusion = GoalGaussianDiffusion(
            model=unet,
            channels=3,
            image_size=self.superres_size,
            timesteps=100,
            sampling_timesteps=10,
            loss_type='l2',
            objective='pred_v',
            beta_schedule = 'cosine',
            min_snr_loss_weight = True,
			guidance_weight=0,
        )

        state_dict = torch.load(model_id, map_location='cpu')
        diffusion.load_state_dict(state_dict)
        self.inpainting_model = diffusion.to(self.device).eval()

    def init_vdm(self, model_id):
        if model_id is None:
            return
        print("Loading topdown VDM model...")
        unet = UnetMaco(embed_dim=4096, num_frames=self.sample_per_seq-1, conds=1)

        diffusion = GoalGaussianDiffusion(
            channels=3*(self.sample_per_seq-1),
            model=unet,
            image_size=self.target_size,
            timesteps=100,
            sampling_timesteps=10,
            loss_type='l2',
            objective='pred_v',
            beta_schedule = 'cosine',
            min_snr_loss_weight = True,
            guidance_weight=self.guidance_weight,
        )

        state_dict = torch.load(model_id, map_location='cpu')
        diffusion.load_state_dict(state_dict)
        self.model = diffusion.to(self.device).eval()

    def init_superres(self, model_id):
        if model_id is None:
            return
        print("Loading super resolution model...")
        unet = UnetSuperRes(target_size=self.superres_size)

        model = GoalGaussianDiffusion(
            model=unet,
            channels=3,
            image_size=self.superres_size,
            timesteps=1000,
            sampling_timesteps=20,
            loss_type='l2',
            objective='pred_v',
            beta_schedule = 'cosine',
            min_snr_loss_weight = True,
        )

        state_dict = torch.load(model_id, map_location='cpu')
        model.load_state_dict(state_dict)
        self.superres_model = model.to(self.device).eval()

    def tensor2list(self, tensor):
        return (tensor.cpu().numpy().clip(0, 1) * 255).astype('uint8').tolist()

    def collect_composed_text_embed(self, text_goal):
        max_agent = 4
        text_embed_list = [[] for _ in range(max_agent)]
        max_len = 0
        for comp_text in text_goal:
            for i, text in enumerate(comp_text):
                embed = self.cache.get(text)
                if embed is None:
                    print(f"Cache miss: {text}\n, using zero embedding")
                    max_len = max(max_len, 1)
                    text_embed_list[i].append(torch.zeros(1, 4096))
                else:
                    max_len = max(max_len, embed.size(0))
                    text_embed_list[i].append(embed)
            for i in range(len(comp_text), max_agent):
                text_embed_list[i].append(None)

        text_embed, mask, comp_mask = [], [], []
        for i in range(max_agent):
            embeds = torch.zeros((len(text_embed_list[i]), max_len, 4096), device=self.device)
            embed_mask = torch.zeros((len(text_embed_list[i]), max_len), dtype=torch.bool, device=self.device)
            comp = torch.zeros(len(text_embed_list[i]), dtype=torch.bool, device=self.device)
            for j, embed in enumerate(text_embed_list[i]):
                if embed is not None:
                    embeds[j, :embed.size(0)] = embed
                    embed_mask[j, :embed.size(0)] = True
                    comp[j] = True
            text_embed.append(embeds)
            mask.append(embed_mask)
            comp_mask.append(comp)
        return text_embed, mask, comp_mask

    def sample(self, text_goal, inpainting_text_goal, x_cond):
        x_cond = torch.tensor(x_cond).to(self.device)
        bs = x_cond.size(0)

        if text_goal[0] is None:
            topdown = self.inpainting_model.sample(x_cond, batch_size=bs)
            return None, None, self.tensor2list(topdown)
        else:
            text_embed, embed_mask, comp_mask = self.collect_composed_text_embed(text_goal)

            output = self.model.sample(x_cond, text_embed, batch_size=bs, mask=embed_mask, comp_mask=comp_mask)
            output = output.reshape(bs, -1, 3, *self.target_size)
            # x_cond = x_cond.reshape(bs, -1, 3, *self.target_size)

            for i in range(bs):
                all_wait = True
                for line in text_goal[i]:
                    line = line.strip()
                    if len(line) > 0 and "wait" not in line:
                        all_wait = False
                        break
                if all_wait:
                    output[i] = repeat(x_cond[i, -3:], 'c h w -> n c h w', n=self.sample_per_seq-1)

            superres_output = self.superres_model.sample(output[:, -1], batch_size=bs)
            return self.tensor2list(output), self.tensor2list(superres_output), None

    def run(self, input_dicts):
        for input_dict in input_dicts:
            # print(input_dict)
            input_dict[0] = Image.open(input_dict[0]).convert('RGB')
            if input_dict[1] is None: # inpainting
                input_dict[0] = self.inpainting_transform(input_dict[0])
            else:
                input_dict[0] = self.transform(input_dict[0])

        output_paths, x_cond, text_goals, inpainting_text_goals = [], [], [], []
        for input_dict in input_dicts:
            conds, text_goal, inpainting_text_goal, output_dir, idx = input_dict
            x_conds = conds.to(self.device)
            x_cond.append(x_conds)
            text_goals.append(text_goal)
            inpainting_text_goals.append(inpainting_text_goal)

        x_cond = torch.stack(x_cond, dim=0)
        output, image, topdown = self.sample(text_goals, inpainting_text_goals, x_cond)

        for i, input_dict in enumerate(input_dicts):
            conds, text_goal, inpainting_text_goal, output_dir, idx = input_dict
            if text_goal is None:
                output_path = os.path.join(output_dir, f'{idx}.png')
                utils.save_image(torch.tensor(topdown[i]).float() / 255, output_path)
                output_paths.append(output_path)
            else:
                conds = conds.view(-1, 3, *self.target_size)
                save_img = torch.concat([conds, torch.tensor(output[i]).float() / 255], dim=0)
                utils.save_image(save_img, os.path.join(output_dir, f"outcome_debug_{idx}.png"), nrow=self.sample_per_seq-1+conds.size(0))
                output_path = os.path.join(output_dir, f'outcome_{idx}.png')
                Image.fromarray(np.array(image[i], dtype=np.uint8).transpose((1, 2, 0))).save(output_path)
                output_paths.append(output_path)
        return output_paths