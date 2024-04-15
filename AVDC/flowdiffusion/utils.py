import os
import glob
import json
from pathlib import Path
import pickle
import shutil
import torch
import torchvision
from PIL import Image
import cv2
import numpy as np
from transformers import T5Tokenizer
import re

def get_paths(root=""):
    f = []
    for dirpath, dirname, filename in os.walk(root):
        if "image" in dirpath:
            f.append(dirpath)
    print(f"Found {len(f)} sequences")
    return f

def get_paths_from_dir(dir_path):
    paths = glob.glob(os.path.join(dir_path, 'im*.jpg'))
    try:
        paths = sorted(paths, key=lambda x: int((x.split('/')[-1].split('.')[0])[3:]))
    except:
        print(paths)
    return paths

class Cache:
    def __init__(self, root="../cache", rebuild=False):
        self.root = Path(root)
        if rebuild:
            if self.root.exists():
                shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        self.total = 0
        if (self.root / "cache.json").exists():
            js = json.loads((self.root / "cache.json").read_text())
            self.cache = js["cache"]
            self.total = js["total"]
    
    def get(self, key):
        if key in self.cache:
            path = self.cache[key]
            return pickle.loads((self.root / path).read_bytes())
        else:
            return None
    
    def set(self, key, value):
        path = f"{self.total}.pk"
        self.cache[key] = path
        self.total += 1
        (self.root / path).write_bytes(pickle.dumps(value))
    
    def save(self):
        (self.root / "cache.json").write_text(json.dumps({"cache": self.cache, "total": self.total}, indent=4))

class Visualizer:
    def __init__(self):
        self.attn_maps = []
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
    
    def add_attn_map(self, attn):
        # attn: (B, F, H, W, T)
        assert attn.shape[0] == 1
        attn = attn.mean(1).permute(0, 3, 1, 2)
        attn = torch.nn.functional.interpolate(attn, (128, 128), mode='bilinear')
        self.attn_maps.append(attn.squeeze(0))
    
    def vis_attn_map(self, prompt, out_file):
        tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(prompt))
        attn = torch.stack(self.attn_maps).mean(0)
        assert attn.shape[0] == len(tokens)

        words = re.split(r"[^a-zA-Z\-_]", prompt)
        total_token, total_len = 0, 0
        word_attn = []
        for word in words:
            if word == "":
                total_len += 1
                continue
            num_token = len(self.tokenizer.tokenize(word))
            print(f"{word}: {tokens[total_token:total_token + num_token]}")
            word_attn.append(attn[total_token:total_token + num_token].mean(0))

            total_len += len(word) + 1
            total_token += num_token
            if total_len - 1 < len(prompt) and prompt[total_len - 1].strip() != "":
                total_token += 1
        assert total_token + 1 == len(tokens)
        word_attn.append(attn[-1])
        attn = torch.stack(word_attn, dim=0)
        attn = attn / attn.max(dim=1)[0].max(dim=1)[0].view(-1, 1, 1)
        attn = (attn.view(-1, 128, 128, 1).repeat(1, 1, 1, 3) * 255).clamp(0, 255).byte().cpu().numpy()

        words = [w for w in words if w != ""]
        words.append("</s>")
        img = np.concatenate([attn, np.zeros((attn.shape[0], 32, 128, 3), dtype=np.uint8)], axis=1)
        img = img.transpose(1, 0, 2, 3).reshape(-1, attn.shape[0] * 128, 3)
        for i, word in enumerate(words):
            cv2.putText(img, word, (128 * i + 16, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        Image.fromarray(img).save(out_file)
        