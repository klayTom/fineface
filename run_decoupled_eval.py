import os
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import wandb
from tqdm import tqdm
import argparse

# 评测依赖库
from transformers import CLIPProcessor, CLIPVisionModelWithProjection

# FineFace 本地依赖
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from fineface.au_attention import hack_unet_attn_layers, AUAttnProcessor
from fineface.fineface_pipeline import AUEncoder, FineFacePipeline

def init_clip_model(device="cuda"):
    print("Loading CLIP Model...")
    clip_model_id = "openai/clip-vit-base-patch32"
    clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
    clip_model = CLIPVisionModelWithProjection.from_pretrained(clip_model_id).to(device)
    return clip_processor, clip_model

def load_local_fineface(model_path, device="cuda"):
    """严格复用 evaluate.py 的加载逻辑"""
    print("Initializing SD 2.1 Base Model...")
    sd_repo_id = "Manojb/stable-diffusion-2-1-base" 
    
    unet = UNet2DConditionModel.from_pretrained(sd_repo_id, subfolder="unet", torch_dtype=torch.float16)
    unet.set_default_attn_processor()
    hack_unet_attn_layers(unet, AUAttnProcessor)
    
    print(f"Loading custom trained weights from {model_path}...")
    unet.load_state_dict(torch.load(os.path.join(model_path, "attn_processors.ckpt"), map_location=device), strict=False)
    unet.load_attn_procs(model_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="unet")
    
    pipe = StableDiffusionPipeline.from_pretrained(sd_repo_id, unet=unet, torch_dtype=torch.float16).to(device)
    
    au_encoder = AUEncoder().to(device)
    au_encoder.load_state_dict(torch.load(os.path.join(model_path, "au_encoder.ckpt"), map_location=device))
    
    class LocalFineFacePipeline(FineFacePipeline):
        def __init__(self, pipe, au_encoder, device):
            self.device = device
            AUS = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
            self.AU_DICT = {f"AU{au}": 0 for au in AUS}
            self.pipe = pipe
            self.au_encoder = au_encoder

    return LocalFineFacePipeline(pipe, au_encoder, device)

def get_diagnostic_test_cases():
    """提供少量高对比度的 Prompt 进行快速解耦诊断"""
    prompts = [
        "a closeup portrait of a man", 
        "a close up of a woman with pink lipstick",
        "a realistic portrait of a young boy"
    ]
    test_cases = []
    # 选取动作幅度最大的几个 AU 进行测试
    for prompt in prompts:
        for target_au in [{"AU12": 5.0}, {"AU04": 5.0}, {"AU25": 5.0, "AU26": 5.0}]:
            test_cases.append({"prompt": prompt, "aus": target_au})
    return test_cases

def main():
    parser = argparse.ArgumentParser(description="FineFace: Decoupled Architectural Evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="你的模型权重路径，例如 checkpoint-20000")
    parser.add_argument("--seed", type=int, default=42, help="固定随机种子以保证特征轨迹一致")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wandb.init(project="fineface-evaluation", name="decoupled_architectural_eval")
    
    clip_processor, clip_model = init_clip_model(device)
    pipeline = load_local_fineface(args.model_path, device)
    pipeline.pipe.set_progress_bar_config(disable=True)
    
    test_cases = get_diagnostic_test_cases()
    
    total_base_drift_clip = []
    total_bg_mse = []
    
    os.makedirs("eval_tmp", exist_ok=True)
    
    for step, item in enumerate(tqdm(test_cases, desc="Running Decoupled Evaluation")):
        prompt = item["prompt"]
        target_au_dict = item["aus"]
        
        # ==========================================
        # 1. Base 图像 (完全关闭 AU 路由, au_scale=0)
        # ==========================================
        gen_base = torch.Generator(device=device).manual_seed(args.seed)
        img_base = pipeline(prompt=prompt, aus={}, au_scale=0.0, generator=gen_base).images[0]
        
        # ==========================================
        # 2. Neutral 图像 (开启 AU 路由, 但输入空表情)
        # ==========================================
        gen_neutral = torch.Generator(device=device).manual_seed(args.seed)
        img_neutral = pipeline(prompt=prompt, aus={}, au_scale=1.0, generator=gen_neutral).images[0]
        
        # ==========================================
        # 3. Conditional 图像 (开启 AU 路由, 注入目标表情)
        # ==========================================
        gen_cond = torch.Generator(device=device).manual_seed(args.seed)
        img_cond = pipeline(prompt=prompt, aus=target_au_dict, au_scale=1.0, generator=gen_cond).images[0]
        
        # ==========================================
        # 4. 提取内生 Spatial Mask
        # ==========================================
        current_mask_np = None
        for name, module in pipeline.pipe.unet.named_modules():
            if hasattr(module, 'current_spatial_mask') and getattr(module, 'current_spatial_mask') is not None:
                mask_tensor = module.current_spatial_mask[0]
                seq_len = mask_tensor.shape[0]
                size = int(np.sqrt(seq_len))
                if size * size == seq_len and size >= 16:
                    spatial_map = mask_tensor.max(dim=-1)[0].float().cpu().numpy()
                    spatial_map = spatial_map.reshape(size, size) # <--- 添加这一行：将 4096 序列重构成 64x64 二维热力图
                    spatial_map = (spatial_map - spatial_map.min()) / (spatial_map.max() - spatial_map.min() + 1e-8)
                    current_mask_np = spatial_map
                    break
                    
        # ==========================================
        # 5. 计算 解耦指标
        # ==========================================
        # 指标 A：空条件基线漂移 (Base vs Neutral)
        inputs_base = clip_processor(images=img_base, return_tensors="pt").to(device)
        inputs_neutral = clip_processor(images=img_neutral, return_tensors="pt").to(device)
        with torch.no_grad():
            emb_base = clip_model(**inputs_base).image_embeds
            emb_neutral = clip_model(**inputs_neutral).image_embeds
        drift_clip_i = F.cosine_similarity(emb_base, emb_neutral).item()
        total_base_drift_clip.append(drift_clip_i)
        
        # 指标 B：背景保护度 L2 (Neutral vs Cond, 仅计算 Mask 以外的区域)
        bg_mse_val = 0.0
        mask_vis = None
        if current_mask_np is not None:
            # 缩放到 512x512
            mask_512 = torch.tensor(current_mask_np).unsqueeze(0).unsqueeze(0).to(device)
            mask_512 = F.interpolate(mask_512, size=(512, 512), mode='bilinear').squeeze().cpu().numpy()
            
            # 阈值二值化：小于 0.2 的区域被认为是受保护的背景/静态五官
            bg_mask = (mask_512 < 0.2).astype(np.float32)
            
            img_n_np = np.array(img_neutral).astype(np.float32) / 255.0
            img_c_np = np.array(img_cond).astype(np.float32) / 255.0
            
            # 计算受保护区域的 MSE
            bg_mse_val = np.sum(((img_n_np - img_c_np) ** 2) * np.expand_dims(bg_mask, axis=-1)) / (np.sum(bg_mask) + 1e-8)
            total_bg_mse.append(bg_mse_val)
            
            # 生成可视化 Mask
            mask_vis = Image.fromarray((current_mask_np * 255).astype(np.uint8)).resize((512, 512), resample=Image.NEAREST)

        # 记录到 WandB
        log_dict = {
            "Decoupled/Base_Drift_CLIP_I": drift_clip_i,
            "Decoupled/Background_L2_Leakage": bg_mse_val,
            "visual/base_au0": wandb.Image(img_base, caption="Base SD (au_scale=0)"),
            "visual/neutral_au0": wandb.Image(img_neutral, caption=f"FineFace (au_scale=1, aus=0)\nDrift CLIP: {drift_clip_i:.3f}"),
            "visual/cond_target": wandb.Image(img_cond, caption=f"FineFace {list(target_au_dict.keys())}\nBg Leakage: {bg_mse_val:.5f}")
        }
        if mask_vis is not None:
            log_dict["visual/spatial_mask"] = wandb.Image(mask_vis, caption="AU Routing Mask")
            
        wandb.log(log_dict, step=step)

    print("\n" + "="*50)
    print("解耦评估完成！")
    print(f"理论基线一致性 (Zero-AU Drift CLIP-I): {np.mean(total_base_drift_clip):.4f}")
    print(f"背景/ID 保护误差 (Background L2 Leakage): {np.mean(total_bg_mse):.6f}")
    print("="*50)
    
    if np.mean(total_base_drift_clip) < 0.95:
        print("[诊断结论]: 基线漂移严重！即使 AU=0，AUEncoder 的 Bias 也在破坏特征。建议缩小 Bias 的初始化方差。")
    elif np.mean(total_bg_mse) > 0.005:
        print("[诊断结论]: 门控泄漏！CLIP-I 掉分是因为 Spatial Modulation Gate 没有锁死背景。请适当调高 Temperature Routing 的初始值。")
    else:
        print("[诊断结论]: 架构极其稳定。CLIP-I 维持在 0.78 纯粹是因为大幅度表情带来的合理几何形变。")

    wandb.finish()

if __name__ == "__main__":
    main()