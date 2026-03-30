import os
import shutil
import torch
from fineface.fineface_pipeline import FineFacePipeline, AUEncoder
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from fineface.au_attention import hack_unet_attn_layers, AUAttnProcessor

# ======================================================
# 填入你 500 步训练实际保存权重的路径！
# ======================================================
MODEL_PATH = "/root/autodl-tmp/fineface/fineface_spatial_id_ours/checkpoint-15000"

def load_local_model(model_path, device="cuda"):
    print(f"正在加载本地训练权重...")
    sd_repo_id = "Manojb/stable-diffusion-2-1-base"
    
    unet = UNet2DConditionModel.from_pretrained(sd_repo_id, subfolder="unet", torch_dtype=torch.float16)
    unet.set_default_attn_processor()
    hack_unet_attn_layers(unet, AUAttnProcessor)
    
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

def save_mask_if_exists(target_name):
    """把生成的临时 mask 图重命名并移到 results 文件夹中"""
    if os.path.exists("temp_spatial_mask.png"):
        shutil.copy("temp_spatial_mask.png", target_name)
        print(f"  -> 成功捕获对应的 Spatial Mask: {target_name}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_local_model(MODEL_PATH, device)
    
    os.makedirs("results", exist_ok=True)
    fixed_seed = 42 
    prompt = "a closeup of a young man in a park"
    
    print("\n--- 开始生成图片与捕获 Mask ---")
    
    # 测试 1: 强烈的微笑 (AU12 + AU6)
    print("生成: 大笑 (AU6 + AU12)...")
    generator = torch.Generator(device=device).manual_seed(fixed_seed)
    aus_smile = {"AU6": 4.0, "AU12": 5.0}
    img_smile = pipe(prompt, aus=aus_smile, generator=generator).images[0]
    img_smile.save("results/test_smile.jpg")
    save_mask_if_exists("results/mask_smile.png") # 立刻抓取 Mask

    # 测试 2: 皱眉 (AU4)
    print("生成: 皱眉 (AU4)...")
    generator = torch.Generator(device=device).manual_seed(fixed_seed)
    aus_frown = {"AU4": 5.0}
    img_frown = pipe(prompt, aus=aus_frown, generator=generator).images[0]
    img_frown.save("results/test_frown.jpg")
    save_mask_if_exists("results/mask_frown.png") # 立刻抓取 Mask
    
    print("\n搞定！去 'results' 文件夹下看看生成的脸和对应的热力图吧！")

if __name__ == "__main__":
    main()