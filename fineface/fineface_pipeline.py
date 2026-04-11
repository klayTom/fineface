from typing import Union, Dict, List, Optional
from typing import Union, Dict, List, Optional
import torch
from torch import nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from huggingface_hub import hf_hub_download
from fineface.au_attention import hack_unet_attn_layers, AUAttnProcessor

class AUEncoder(torch.nn.Module):
    def __init__(self, number_of_aus: int = 12, hidden_dim: int = 128, clip_dim: int = 1024, pad_zeros: bool = True):
        super().__init__()
        self.n_aus = number_of_aus
        self.clip_dim = clip_dim
        
        # 1. 独立局部特征提取：杜绝不同 AU 输入相同强度时生成相同特征
        self.initial_mapping = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(number_of_aus)
        ])
        
        # 2. 构建先验解剖学邻接矩阵 (基于 FACS 系统)
        # 索引对照: 0:AU1, 1:AU2, 2:AU4, 3:AU5, 4:AU6, 5:AU9, 6:AU12, 7:AU15, 8:AU17, 9:AU20, 10:AU25, 11:AU26
        A = torch.eye(number_of_aus)
        edges = [
            (0, 1), (0, 2), (1, 2),        # 额头与眉毛联动
            (3, 4), (4, 5),                # 眼部与鼻部联动
            (6, 4),                        # 笑容联动(AU12颧骨肌与AU6眼轮匝肌)
            (6, 10), (7, 10), (8, 7),      # 嘴角与下巴联动
            (9, 10), (10, 11)              # 嘴唇伸展与下颌联动
        ]
        for i, j in edges:
            A[i, j] = 1.0
            A[j, i] = 1.0
            
        # 归一化邻接矩阵
        D_inv_sqrt = torch.diag(torch.sum(A, dim=1) ** -0.5)
        adj_matrix = torch.mm(torch.mm(D_inv_sqrt, A), D_inv_sqrt)
        self.register_buffer("adj_matrix", adj_matrix) 
        
        # 3. GCN 图卷积层
        self.gcn1 = nn.Linear(hidden_dim, hidden_dim)
        self.gcn2 = nn.Linear(hidden_dim, clip_dim)
        
       # 【加大初始扰动方差】
        nn.init.normal_(self.gcn2.weight, mean=0.0, std=0.1) 
        nn.init.zeros_(self.gcn2.bias)
        
        # 专属身份偏置初始化 (给 1e-4 的扰动)
        self.au_embeds = nn.Parameter(torch.randn(number_of_aus, clip_dim) * 1e-4)

    def forward(self, x):
        # x 形状: (Batch, 12)
        x_expanded = x.unsqueeze(-1) 
        
        # 独立映射
        features = [self.initial_mapping[i](x_expanded[:, i, :]) for i in range(self.n_aus)]
        h = torch.stack(features, dim=1) # (Batch, 12, hidden_dim)
        
        # 图信息传递
        h = torch.matmul(self.adj_matrix, h) 
        h = F.silu(self.gcn1(h))
        h = torch.matmul(self.adj_matrix, h)
        h = self.gcn2(h) # (Batch, 12, 1024)
        
        # 残差连接固有特征
        out = h + self.au_embeds.unsqueeze(0)
        
        # 注意：现在输出维度是 (Batch, 12, 1024) 也就是 12 个 Token，而不是原来的 1 个 Token
        return out


class FineFacePipeline:
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        sd_repo_id = "stabilityai/stable-diffusion-2-1-base"
        fineface_repo_id = "Tvaranka/fineface"
        AUS = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
        self.AU_DICT = {f"AU{au}": 0 for au in AUS}

        unet = UNet2DConditionModel.from_pretrained(sd_repo_id, subfolder="unet", cache_dir=cache_dir)
        unet.set_default_attn_processor()
        hack_unet_attn_layers(unet, AUAttnProcessor)
        unet.load_state_dict(
            torch.load(hf_hub_download(fineface_repo_id, "attn_processors.ckpt"), map_location=self.device),
            strict=False
        )
        unet.load_attn_procs(
            fineface_repo_id, weight_name="pytorch_lora_weights.safetensors", adapter_name="unet"
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(sd_repo_id, unet=unet, cache_dir=cache_dir).to(self.device)

        self.au_encoder = AUEncoder().to(self.device)
        self.au_encoder.load_state_dict(
            torch.load(hf_hub_download(fineface_repo_id, "au_encoder.ckpt"), map_location=self.device)
        )

    def encode_aus(self, aus: Union[Dict, List[Dict]] = None):
        if isinstance(aus, Dict):
            aus = [aus]
        # Create empty dict with all AUs
        new_aus_dict = [self.AU_DICT.copy() for _ in aus]
        # Update created dicts with AU prompts
        [new_aus_dict[i].update(aus[i]) for i in range(len(aus))]
        new_aus = [list(au_dict.values()) for au_dict in new_aus_dict]
        new_aus = torch.tensor(new_aus).float().to(self.device)
        # Encode AUs
        au_embeds = self.au_encoder(new_aus)
        uncond_aus = torch.zeros_like(new_aus)
        negative_au_embeds = self.au_encoder(uncond_aus)
        return torch.cat([negative_au_embeds, au_embeds], dim=0)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        aus: Union[Dict, List[Dict]] = None,
        au_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):

        au_embeds = self.encode_aus(aus).to(self.device)
        output = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            cross_attention_kwargs={"au_embedding": au_embeds, "au_scale": au_scale},
        )
        return output
        

        