from goal_diffusion import GoalGaussianDiffusion, Trainer, TDWMacoTopdownTrainer, to_device
from unet import UnetMaco as Unet, UnetTDWMacoInpainting
from transformers import T5Tokenizer, T5EncoderModel
from datasets import TDWMacoDataset
from torch.utils.data import Subset, DataLoader
import argparse
from pathlib import Path
from torchvision import utils
from einops import rearrange
from tqdm import tqdm
import torch
from PIL import Image
from torchvision import transforms
import imageio
from utils import Cache
from functools import partial

class TDWMacoTrainer:
    def __init__(self, args):
        self.mode = args.mode
        self.args = args
        self.target_size = (336, 336) if args.inpainting else (128, 128)
        self.sample_per_seq = 8
        self.valid_n = 8
        self.train_path = {
        }
        self.test_path = {
        }
        
        self.result_path = "../results/tdw_maco"
        if args.inpainting:
            self.result_path += "_inpainting"
        elif args.single:
            self.result_path += "_single"
        else:
            self.result_path += "_multiple"
        
        if args.result_dir is not None:
            self.result_path = args.result_dir
        self.conds = 1

        if self.mode == 'preprocess':
            self.cache = Cache()
        else:
            self.cache = Cache()
        
        self.train_set = self.valid_set = self.test_set = [None]
        if self.mode != 'inference' and self.mode != 'get_model':
            self.test_set = TDWMacoDataset(
                sample_per_seq=self.sample_per_seq,
                paths=self.test_path,
                target_size=self.target_size,
                task_name=args.task_name,
                embed_cache=(None if self.mode == 'preprocess' else self.cache),
                inpainting=args.inpainting,
                single=args.single,
            )
            if self.mode != 'test':
                self.train_set = TDWMacoDataset(
                    sample_per_seq=self.sample_per_seq,
                    paths=self.train_path,
                    target_size=self.target_size,
                    task_name=args.task_name,
                    embed_cache=(None if self.mode == 'preprocess' else self.cache),
                    inpainting=args.inpainting,
                    single=args.single,
                )
                if self.mode == 'train':
                    self.valid_inds = [i for i in range(0, len(self.test_set), len(self.test_set)//self.valid_n)][:self.valid_n]
                    self.valid_set = Subset(self.test_set, self.valid_inds)
        
        if self.mode == 'preprocess':
            self.init_lm()
            self.preprocess(self.train_set)
            self.preprocess(self.test_set)
            return
        
        self.init_trainer()

        if args.checkpoint_num is not None:
            self.trainer.load(args.checkpoint_num)
    
        if args.mode == 'train':
            self.trainer.train()
            return
        elif args.mode == 'get_model':
            torch.save(self.trainer.ema.ema_model.state_dict(), Path(self.result_path, "maco_model.pt"))
            return
        
        if not args.no_superres:
            self.superres_model = self.trainer.accelerator.prepare(torch.load("../results/super_res/super_res_model.pt").eval())
        if args.mode == 'test':
            self.test(Path(self.result_path, "test"))
        elif args.mode == 'test_train':
            self.test_set = self.train_set
            self.test(Path(self.result_path, "test_train"))
    
    def init_lm(self):
        if hasattr(self, 'tokenizer'):
            return
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(self.args.lm_id)
        self.text_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(self.args.lm_id)
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
    
    def init_trainer(self):
        if self.args.inpainting:
            unet = UnetTDWMacoInpainting(embed_dim=512 if self.args.debug else 4096, conds=self.conds)

            self.diffusion = GoalGaussianDiffusion(
                model=unet,
                channels=3,
                image_size=self.target_size,
                timesteps=1 if self.args.debug else 100,
                sampling_timesteps=1 if self.args.debug else 10,
                loss_type='l2',
                objective='pred_v',
                beta_schedule = 'cosine',
                min_snr_loss_weight = True,
            )

            self.trainer = TDWMacoTopdownTrainer(
                diffusion_model=self.diffusion,
                train_set=self.train_set,
                valid_set=self.valid_set,
                train_lr=1e-4,
                train_num_steps = 100000,
                save_and_sample_every = 2 if self.args.debug else 1000,
                ema_update_every = 10,
                ema_decay = 0.999,
                train_batch_size = 1 if self.args.debug else 96,
                valid_batch_size = 32,
                gradient_accumulate_every = 1,
                num_samples=self.valid_n, 
                results_folder=self.result_path,
                fp16 =True,
                amp=True,
                save_milestone=args.save_milestone,
                calculate_fid=False,
            )
        else:
            bs = 1 if self.args.debug else (384 if self.args.single else 120)
            unet = Unet(embed_dim=512 if self.args.debug else 4096, num_frames=self.sample_per_seq-1, conds=self.conds)

            self.diffusion = GoalGaussianDiffusion(
                channels=3*(self.sample_per_seq-1),
                model=unet,
                image_size=self.target_size,
                timesteps=1 if self.args.debug else 100,
                sampling_timesteps=1 if self.args.debug else 10,
                loss_type='l2',
                objective='pred_v',
                beta_schedule = 'cosine',
                min_snr_loss_weight = True,
                guidance_weight = self.args.guidance_weight,
            )

            self.trainer = Trainer(
                diffusion_model=self.diffusion,
                channels=3,
                tokenizer=None, 
                text_encoder=None,
                train_set=self.train_set,
                valid_set=self.valid_set,
                train_lr=1e-4,
                train_num_steps = 200000,
                save_and_sample_every = 2 if self.args.debug else 1000,
                ema_update_every = 10,
                ema_decay = 0.999,
                train_batch_size = 1 if self.args.debug else bs,
                valid_batch_size = 32,
                gradient_accumulate_every = 1,
                num_samples=self.valid_n, 
                results_folder = self.result_path,
                fp16 =True,
                amp=True,
                save_milestone=self.args.save_milestone,
                calculate_fid=False,
                embed_preprocessed=True,
                composed=not self.args.inpainting and not self.args.single,
            )
    
    def preprocess(self, dataset: TDWMacoDataset):
        self.text_encoder = self.text_encoder.cuda()
        for i in tqdm(range(len(dataset)), desc='Preprocessing'):
            text, task_name = dataset[i]
            if len(text) > 0 and isinstance(text[0], tuple):
                n_text = []
                for t in text:
                    n_text.extend(t)
                text = n_text
            for t in text:
                if self.cache.get(t) is None:
                    text_ids = self.tokenizer([t], return_tensors = 'pt', padding = True, truncation = True, max_length = 128).to(self.text_encoder.device)
                    result = self.text_encoder(**text_ids).last_hidden_state.cpu().squeeze(0)
                    self.cache.set(t, result)
                    if result.isnan().any():
                        raise ValueError("text_embed has nan")
        self.cache.save()
    
    def test(self, output_dir):
        output_dir.mkdir(exist_ok=True)
        loader = DataLoader(self.test_set, batch_size=6, shuffle=True, num_workers=4,
            collate_fn=partial(Trainer.embed_collect_fn, composed=not self.args.inpainting and not self.args.single))
        loader = self.trainer.accelerator.prepare(loader)
        for batch in tqdm(loader, disable = not self.trainer.accelerator.is_main_process, desc='Testing'):
            x_gt, x_cond, tup, text_goal, task_name = to_device(batch, self.trainer.accelerator.device)
            f = 1 if self.args.inpainting else self.sample_per_seq - 1
            x_gt = rearrange(x_gt, "b (f c) h w -> b f c h w", f=f)
            if not self.args.inpainting:
                text_embed, embed_mask, comp_mask = tup
            else:
                text_embed, embed_mask = tup
                comp_mask = None
            self.sample(text_goal, task_name, text_embed, embed_mask, comp_mask, x_cond, output_dir, x_gt)
    
    def inference(self, text_goal, image_path, output_dir, return_all_timesteps = False, vis = None):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        if not isinstance(text_goal, list):
            text_goal = [text_goal]
        
        text_embed, embed_mask, comp_mask = [], [], []
        for t in text_goal:
            embed = self.cache.get(t)
            if embed is None:
                print(f"Error: Cache miss for... {t}")
                exit(-1)
                # self.init_lm()
                # self.text_encoder = self.text_encoder.to(self.trainer.accelerator.device)
                # text_ids = self.tokenizer([text_goal], return_tensors = 'pt', padding = True, truncation = True, max_length = 128).to(self.trainer.accelerator.device)
                # text_embed = self.text_encoder(**text_ids).last_hidden_state
                # self.cache.set(text_goal, text_embed.squeeze(0).cpu())
                # self.cache.save()
            else:
                embed = embed.unsqueeze(0).to(self.trainer.accelerator.device)
                text_embed.append(embed)
                embed_mask.append(torch.ones((1, embed.size(1)), dtype=torch.bool).to(self.trainer.accelerator.device))
                comp_mask.append(torch.ones((1,), dtype=torch.bool).to(self.trainer.accelerator.device))
        
        image = Image.open(image_path)
        transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
        ])
        image = transform(image)
        x_cond = image.unsqueeze(0).to(self.trainer.accelerator.device)
        self.sample([text_goal], [None], text_embed, embed_mask, comp_mask, x_cond, output_dir, return_all_timesteps = return_all_timesteps, vis = vis)
    
    def sample(self, text_goal, task_name, text_embed, embed_mask, comp_mask, x_cond, output_root, x_gt = None, return_all_timesteps = False, vis = None):
        bs = x_cond.size(0)
        tup = [text_embed, embed_mask] + ([comp_mask] if not self.args.inpainting else [])
        output = self.trainer.sample(x_cond, tup, bs, return_all_timesteps, vis)
        if return_all_timesteps:
            self.save_images(output.reshape(-1, 3, *self.target_size).cpu(), output_root / "all_timestep.png", nrow=self.sample_per_seq-1)
            output = output[:, -1]
        output = output.reshape(bs, -1, 3, *self.target_size)
        x_cond = x_cond.reshape(bs, -1, 3, *self.target_size)
        if x_gt is not None:
            x_gt = x_gt.reshape(bs, -1, 3, *self.target_size)
        output = torch.cat([x_cond, output], dim=1)

        if not self.args.no_superres:
            superres_output = []
            for i in range(bs):
                superres_output.append(self.superres_model.sample(output[i], batch_size=output.shape[1]))
            superres_output = torch.stack(superres_output, dim=0)
        
        for i, task in enumerate(task_name):
            output_dir = output_root / task if task is not None else output_root
            output_dir.mkdir(exist_ok=True, parents=True)
            (output_dir / "goal.txt").write_text(str(text_goal[i]))

            if x_gt is not None:
               self.save_images(torch.cat([x_cond[i], x_gt[i]], dim=0).cpu(), output_dir / "ground_truth.png")
            self.save_images(output[i].cpu(), output_dir / "output.png")
            cur_output = (rearrange(output[i].cpu().numpy(), "f c h w -> f h w c").clip(0, 1) * 255).astype('uint8')
            imageio.v2.mimsave(output_dir / "output.gif", cur_output, duration=1000, loop=5)
            imageio.v2.mimsave(output_dir / "output.mp4", cur_output, fps=1, quality=10)

            if not self.args.no_superres:
                self.save_images(superres_output[i].cpu(), output_dir / "superres_output.png")
                cur_superres_output = (rearrange(superres_output[i].cpu().numpy(), "f c h w -> f h w c").clip(0, 1) * 255).astype('uint8')
                imageio.v2.mimsave(output_dir / "superres_output.gif", cur_superres_output, duration=1000, loop=5)
                imageio.v2.mimsave(output_dir / "superres_output.mp4", cur_superres_output, fps=1, quality=10)
            print(f'Generated {output_dir}')
    
    def save_images(self, images, path, nrow=None): # [f c h w]
        if nrow is None:
            nrow = self.sample_per_seq+self.conds-1
        utils.save_image(images, path, nrow=nrow)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='train', choices=['train', 'test', 'inference', 'preprocess', 'test_train', 'get_model']) # set to 'inference' to generate samples
    parser.add_argument('-c', '--checkpoint_num', type=int, default=None, help="-1 for use model_recent.pt") # set to checkpoint number to resume training or generate samples
    parser.add_argument('-p', '--inference_path', type=str, default=None, help="path to the image of the first frame") # set to path to generate samples
    parser.add_argument('-t', '--text_path', type=str, default=None, help="path to the text instruction file") # set to text to generate samples
    parser.add_argument('--lm_id', type=str, default='google/t5-v1_1-xxl', help="language model id for the encoder")
    parser.add_argument('--result_dir', type=str, default=None, help="path to the result directory")
    parser.add_argument('--guidance_weight', type=float, default=0)
    parser.add_argument('--save_milestone', action="store_true")
    parser.add_argument('--single', action="store_true")
    parser.add_argument('--inpainting', action="store_true")
    parser.add_argument('--no_superres', action="store_true")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--task_name', nargs="*", default=[])
    args = parser.parse_args()
    if args.mode == 'inference' or args.mode == 'test' or args.mode == 'test_train' or args.mode == 'get_model':
        assert args.checkpoint_num is not None
    maco = TDWMacoTrainer(args)
    if args.mode == 'inference':
        maco.inference(Path(args.text_path).read_text(), args.inference_path, Path("../results/maco/inference"))
