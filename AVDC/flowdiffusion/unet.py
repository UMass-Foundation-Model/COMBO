import sys
from pathlib import Path

# Get the current directory of file.py
current_dir = Path(__file__).resolve().parent

# Add the current directory to sys.path
sys.path.append(str(current_dir))

from guided_diffusion.unet import UNetModel, SuperResModel
from torch import nn
import torch
from einops import repeat, rearrange


class UnetBridge(nn.Module):
    def __init__(self):
        super(UnetBridge, self).__init__()

        self.unet = UNetModel(
            image_size=(48, 64),
            in_channels=6,
            model_channels=160,
            out_channels=3,
            num_res_blocks=3,
            attention_resolutions=(4, 8),
            dropout=0,
            channel_mult=(1, 2, 4),
            conv_resample=True,
            dims=3,
            num_classes=None,
            task_tokens=True,
            task_token_channels=512,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
        )
        self.unet.convert_to_fp32()

    def forward(self, x, x_cond, t, task_embed=None, **kwargs):
        f = x.shape[1] // 3
        x_cond = repeat(x_cond, 'b c h w -> b c f h w', f=f)
        x = rearrange(x, 'b (f c) h w -> b c f h w', c=3)
        x = torch.cat([x, x_cond], dim=1)
        out = self.unet(x, t, task_embed, **kwargs)
        return rearrange(out, 'b c f h w -> b (f c) h w')

class UnetMW(nn.Module):
    def __init__(self):
        super(UnetMW, self).__init__()
        self.unet = UNetModel(
            image_size=(128, 128),
            in_channels=6,
            model_channels=128,
            out_channels=3,
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            dropout=0,
            channel_mult=(1, 2, 3, 4, 5),
            conv_resample=True,
            dims=3,
            num_classes=None,
            task_tokens=True,
            task_token_channels=512,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
        )
    def forward(self, x, x_cond, t, task_embed=None, **kwargs):
        f = x.shape[1] // 3
        x_cond = repeat(x_cond, 'b c h w -> b c f h w', f=f)
        x = rearrange(x, 'b (f c) h w -> b c f h w', c=3)
        x = torch.cat([x, x_cond], dim=1)
        out = self.unet(x, t, task_embed, **kwargs)
        return rearrange(out, 'b c f h w -> b (f c) h w')
      
class UnetMW_flow(nn.Module):
    def __init__(self):
        super(UnetMW_flow, self).__init__()
        self.unet = UNetModel(
            image_size=(128, 128),
            in_channels=5,
            model_channels=128,
            out_channels=2,
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            dropout=0,
            channel_mult=(1, 2, 3, 4, 5),
            conv_resample=True,
            dims=3,
            num_classes=None,
            task_tokens=True,
            task_token_channels=512,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
        )
    def forward(self, x, x_cond, t, task_embed=None, **kwargs):
        f = x.shape[1] // 2
        x_cond = repeat(x_cond, 'b c h w -> b c f h w', f=f)
        x = rearrange(x, 'b (f c) h w -> b c f h w', c=3)
        x = torch.cat([x, x_cond], dim=1)
        out = self.unet(x, t, task_embed, **kwargs)
        return rearrange(out, 'b c f h w -> b (f c) h w')
    
class UnetThor(nn.Module):
    def __init__(self):
        super(UnetThor, self).__init__()

        self.unet = UNetModel(
            image_size=(64, 64),
            in_channels=6,
            model_channels=128,
            out_channels=3,
            num_res_blocks=3,
            attention_resolutions=(4, 8),
            dropout=0,
            channel_mult=(1, 2, 4),
            conv_resample=True,
            dims=3,
            num_classes=None,
            task_tokens=True,
            task_token_channels=512,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
        )
        self.unet.convert_to_fp32()

    def forward(self, x, x_cond, t, task_embed=None, **kwargs):
        f = x.shape[1] // 3
        x_cond = repeat(x_cond, 'b c h w -> b c f h w', f=f)
        x = rearrange(x, 'b (f c) h w -> b c f h w', c=3)
        x = torch.cat([x, x_cond], dim=1)
        out = self.unet(x, t, task_embed, **kwargs)
        return rearrange(out, 'b c f h w -> b (f c) h w')
    

class UnetMaco(nn.Module):
    def __init__(self, embed_dim=2048, num_frames=8, conds=1):
        super(UnetMaco, self).__init__()
        self.unet = UNetModel(
            image_size=(128, 128),
            in_channels=3*(conds+1),
            model_channels=128,
            out_channels=3,
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            dropout=0,
            channel_mult=(1, 2, 3, 4, 5),
            conv_resample=True,
            dims=3,
            num_classes=None,
            task_tokens=True,
            task_token_channels=embed_dim,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
            num_frames=num_frames
        )
    def forward(self, x, x_cond, t, task_embed=None, **kwargs):
        f = x.shape[1] // 3
        x_cond = repeat(x_cond, 'b c h w -> b c f h w', f=f)
        x = rearrange(x, 'b (f c) h w -> b c f h w', f=f)
        x = torch.cat([x, x_cond], dim=1)
        out = self.unet(x, t, task_embed, **kwargs)
        return rearrange(out, 'b c f h w -> b (f c) h w')

class UnetTDWMacoInpainting(nn.Module):
    def __init__(self, embed_dim=2048, conds=1):
        super().__init__()
        self.unet = UNetModel(
            image_size=(336, 336),
            in_channels=3*(conds+1),
            model_channels=64,
            out_channels=3,
            num_res_blocks=2,
            attention_resolutions=(16,),
            dropout=0,
            channel_mult=(1, 2, 4, 6, 8),
            conv_resample=True,
            dims=2,
            num_classes=None,
            task_tokens=None,
            task_token_channels=embed_dim,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32
        )
    def forward(self, x, x_cond, t, *args, **kwargs):
        x = torch.cat([x, x_cond], dim=1)
        out = self.unet(x, t, *args, **kwargs)
        return out

class UnetSuperRes(nn.Module):
    def __init__(self, target_size=(512, 512)) -> None:
        super().__init__()
        self.unet = SuperResModel(
            image_size=target_size,
            in_channels=3,
            model_channels=64,
            out_channels=3,
            num_res_blocks=2,
            attention_resolutions=(16,),
            dropout=0,
            channel_mult=(1, 2, 3, 4, 5),
            conv_resample=True,
            dims=2,
            num_classes=None,
            task_tokens=None,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
        )
    
    def forward(self, x, x_cond, t, *args, **kwargs):
        # x: [b 3 nh nw], t: [b], low_res: [b 3 h w]
        return self.unet(x, t, x_cond)