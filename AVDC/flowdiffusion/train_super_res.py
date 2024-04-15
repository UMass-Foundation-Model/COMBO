from goal_diffusion import GoalGaussianDiffusion, SuperResTrainer
from unet import UnetSuperRes as Unet
from datasets import SuperResDataset
from torch.utils.data import Subset
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

def main(args):
    org_size = (128, 128)
    target_size = (336, 336)
    valid_n = 16
    
    if args.mode == "train" or args.mode == "test":
        train_set = SuperResDataset(path="../datasets/tdw_maco/train", org_size=org_size)
        test_set = SuperResDataset(path="../datasets/tdw_maco/test", org_size=org_size)
        valid_inds = [i for i in range(0, len(test_set), len(test_set)//valid_n)][:valid_n]
        valid_set = Subset(train_set, valid_inds)
    else: # todo: inference
        train_set = valid_set = [None]
    
    unet = Unet(target_size=target_size)
    
    model = GoalGaussianDiffusion(
        model=unet,
        channels=3,
        image_size=target_size,
        timesteps=1000,
        sampling_timesteps=20,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )
    
    trainer = SuperResTrainer(
        diffusion_model=model,
        train_set=train_set,
        valid_set=valid_set,
        train_lr=1e-4,
        train_num_steps =100000,
        save_and_sample_every =1000,
        ema_update_every = 10,
        ema_decay = 0.999,
        train_batch_size =12,
        valid_batch_size =32,
        gradient_accumulate_every = 1,
        num_samples=valid_n, 
        results_folder ='../results/super_res',
        fp16 =True,
        amp=True,
        save_milestone=args.save_milestone,
        calculate_fid=False,
    )
    
    if args.checkpoint_num is not None:
        trainer.load(args.checkpoint_num)
    
    if args.mode == "train":
        trainer.train()
    elif args.mode == "get_model":
        torch.save(trainer.ema.ema_model.state_dict(), "../results/super_res/super_res_model.pt")
    elif args.mode == "inference":
        img = Image.open(args.inference_path)
        trans = transforms.Compose([transforms.Resize(org_size), transforms.ToTensor()])
        img = trans(img).to(trainer.device).unsqueeze(0)
        sup = trainer.ema.ema_model.sample(img, batch_size=1)[0]
        sup = (sup.cpu().detach().numpy().clip(0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
        Image.fromarray(sup).save("../results/super_res/super_res_inference.png")
    else: # todo: add test and inference
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='train', choices=['train', 'test', 'inference', 'get_model']) # set to 'inference' to generate samples
    parser.add_argument('-c', '--checkpoint_num', type=int, default=None, help="-1 for use model_recent.pt") # set to checkpoint number to resume training or generate samples
    parser.add_argument('-p', '--inference_path', type=str, default=None, help="path to the image of the first frame") # set to path to generate samples
    parser.add_argument('--save_milestone', action="store_true")
    parser.add_argument('--task_name', nargs="*", default=[])
    args = parser.parse_args()
    if args.mode == 'inference' or args.mode == 'test' or args.mode == 'get_model':
        assert args.checkpoint_num is not None
    main(args)