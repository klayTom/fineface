import os
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import wandb
from tqdm import tqdm

# 评测依赖库
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
from feat import Detector

# FineFace 依赖 (请确保你在包含 fineface 文件夹的同级目录下运行此脚本)
from fineface.fineface_pipeline import FineFacePipeline
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from fineface.au_attention import hack_unet_attn_layers, AUAttnProcessor
from fineface.fineface_pipeline import AUEncoder, FineFacePipeline

# ==========================================
# 用户配置区 (请根据 AutoDL 实际路径修改)
# ==========================================
# 你的模型权重最终保存路径
MODEL_WEIGHT_PATH = "/root/autodl-tmp/fineface/fineface_spatial_id_ours/checkpoint-20000"  
NUM_AUS = 12  

# 严格对齐 au_dataset.py 中的 [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
TARGET_AU_NAMES = [
    "AU01", "AU02", "AU04", "AU05", 
    "AU06", "AU09", "AU12", "AU15", 
    "AU17", "AU20", "AU25", "AU26"
]

def init_eval_models(device="cuda"):
    """初始化用于评测的基础模型"""
    print("Loading CLIP Model (请确保终端已执行 export HF_ENDPOINT=https://hf-mirror.com)...")
    clip_model_id = "openai/clip-vit-base-patch32"
    clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
    clip_model = CLIPVisionModelWithProjection.from_pretrained(clip_model_id).to(device)
    
    print("Loading Py-Feat AU Detector...")
    au_detector = Detector(device=device)
    
    return clip_processor, clip_model, au_detector

def get_1650_test_cases():
    """严格按照论文 Testing details 构建 1650 个合成测试用例"""
    # 2 
    prompts = [
         "a clear frontal face photo", "a closeup portrait of a man", "a closeup portrait of a woman",
        "a professional id photo", "a woman with a bloody face and glasses", "a close up of a person wearing a suit and tie",
        "a close up of a person wearing a fur collar", "a realistic portrait of a young boy", "a close up of a person wearing a hat",
        "a photo of a young girl", "a close up of a person wearing a jacket", "a close up of a woman with blue eyes",
        "a close up of a child's face with blue eyes", "a symmetrical face photo", "a black and white photo of a woman's face"
    ]
    
    test_cases = []
    
    MUTUALLY_EXCLUSIVE_GROUPS = [
        {6, 7},   # AU12 (微笑) vs AU15 (嘴角下拉) 
        {1, 2},   # AU2 (外侧眉毛上抬) vs AU4 (眉毛下压)
    ]
    
    def is_valid_combination(indices):
        idx_set = set(indices)
        for group in MUTUALLY_EXCLUSIVE_GROUPS:
            if len(idx_set.intersection(group)) > 1:
                return False
        return True

    for prompt in prompts:
        # 1. 单独 AU 测试: 12种AU * 5种强度 = 60个用例/Prompt
        intensities = [1.0, 2.0, 3.0, 4.0, 5.0]
        for au_idx in range(NUM_AUS):
            for intensity in intensities:
                au_tensor = torch.zeros((1, NUM_AUS), dtype=torch.float16)
                au_tensor[0, au_idx] = intensity
                test_cases.append({
                    "prompt": prompt,
                    "target_au": au_tensor,
                    "type": f"single_AU_idx{au_idx}_int{intensity}"
                })
                
        # 2. 组合 AU 测试: 每种 Prompt 随机生成 50 组合法的 AU
        for i in range(50):
            au_tensor = torch.zeros((1, NUM_AUS), dtype=torch.float16)
            num_active = np.random.randint(2, 5) 
            
            while True:
                active_indices = np.random.choice(NUM_AUS, num_active, replace=False)
                if is_valid_combination(active_indices):
                    break
                    
            for idx in active_indices:
                au_tensor[0, idx] = np.round(np.random.uniform(1.0, 5.0), 1)
                
            test_cases.append({
                "prompt": prompt,
                "target_au": au_tensor,
                "type": f"combo_random_{i}"
            })
            
    return test_cases

def load_local_fineface(model_path, device="cuda"):
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

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wandb.init(project="fineface-evaluation", name="fineface_strict_eval_1650")
    
    clip_processor, clip_model, au_detector = init_eval_models(device)
    print("Loading Custom FineFace Pipeline...")
    pipeline = load_local_fineface(MODEL_WEIGHT_PATH, device)
    pipeline.pipe.set_progress_bar_config(disable=True)
    
    test_cases = get_1650_test_cases()
    print(f"Total test cases to generate: {len(test_cases)}")
    
    # ==========================================
    # 分组统计变量
    # ==========================================
    single_clip_i, single_au_mse, single_valid = 0, 0, 0
    combo_clip_i, combo_au_mse, combo_valid = 0, 0, 0
    total_clip_i, total_au_mse, total_valid = 0, 0, 0
    
    os.makedirs("eval_tmp", exist_ok=True)
    fixed_seed = 42 
    
    for step, item in enumerate(tqdm(test_cases, desc="Evaluating")):
        prompt = item["prompt"]
        target_au_tensor = item["target_au"].to(device)
        is_single = "single" in item["type"] # 判断当前是单独 AU 还是组合 AU
        
        PIPELINE_AU_KEYS = ["AU1", "AU2", "AU4", "AU5", "AU6", "AU9", "AU12", "AU15", "AU17", "AU20", "AU25", "AU26"]
        target_au_dict = {key: target_au_tensor[0, idx].item() for idx, key in enumerate(PIPELINE_AU_KEYS) if target_au_tensor[0, idx].item() > 0}

        generator_neutral = torch.Generator(device=device).manual_seed(fixed_seed)
        img_neutral = pipeline(prompt=prompt, aus={}, generator=generator_neutral).images[0]
        
        generator_cond = torch.Generator(device=device).manual_seed(fixed_seed)
        img_cond = pipeline(prompt=prompt, aus=target_au_dict, generator=generator_cond).images[0]
        
        # ==========================================
        # 捞取网络层内部的热力图 (Spatial Mask)
        # ==========================================
        current_mask = None
        # 【修改这里】：遍历 unet 内部真正的模块 (named_modules)，而不是处理器
        for name, module in pipeline.pipe.unet.named_modules():
            if hasattr(module, 'current_spatial_mask') and getattr(module, 'current_spatial_mask') is not None:
                mask_tensor = module.current_spatial_mask[0] 
                seq_len = mask_tensor.shape[0]
                size = int(np.sqrt(seq_len))
                
                # 寻找合适分辨率的层（提取 16x16 或以上的分辨率）
                if size * size == seq_len and size >= 16:
                    spatial_map = mask_tensor.max(dim=-1)[0].float().cpu().numpy()
                    spatial_map = (spatial_map - spatial_map.min()) / (spatial_map.max() - spatial_map.min() + 1e-8)
                    spatial_map = (spatial_map * 255).astype(np.uint8)
                    
                    current_mask = Image.fromarray(spatial_map).resize((512, 512), resample=Image.NEAREST)
                    break

        # 计算 CLIP-I
        inputs_neutral = clip_processor(images=img_neutral, return_tensors="pt").to(device)
        inputs_cond = clip_processor(images=img_cond, return_tensors="pt").to(device)
        with torch.no_grad():
            emb_neutral = clip_model(**inputs_neutral).image_embeds
            emb_cond = clip_model(**inputs_cond).image_embeds
        clip_i = F.cosine_similarity(emb_neutral, emb_cond).item()
        
        # 计算 AU MSE
        tmp_path = f"eval_tmp/cond_{step}.jpg"
        img_cond.save(tmp_path)
        prediction = au_detector.detect_image(tmp_path)
        
        if len(prediction) > 0:
            try:
                pred_aus = prediction[TARGET_AU_NAMES].iloc[0].values.astype(np.float32)
                
                if np.isnan(pred_aus).any():
                    print(f"\n[提示] Step {step}: 未检测到有效人脸，跳过指标计算。")
                else:
                    pred_au_tensor = torch.tensor(pred_aus, dtype=torch.float32) * 5.0
                    target_au_cpu = target_au_tensor.squeeze(0).float().cpu()
                    au_mse = F.mse_loss(pred_au_tensor, target_au_cpu).item()
                    
                    # 归入各自的统计池
                    total_clip_i += clip_i
                    total_au_mse += au_mse
                    total_valid += 1
                    
                    if is_single:
                        single_clip_i += clip_i
                        single_au_mse += au_mse
                        single_valid += 1
                        prefix = "eval_single"
                    else:
                        combo_clip_i += clip_i
                        combo_au_mse += au_mse
                        combo_valid += 1
                        prefix = "eval_combo"

                    # 每一步的结果分别推送到 wandb 的 single 和 combo 面板
                    log_dict = {
                        f"{prefix}/step_clip_i": clip_i,
                        f"{prefix}/step_au_mse": au_mse,
                    }
                    
                    if step < 5 or step % 50 == 0:
                        log_dict["visual/neutral_Y0"] = wandb.Image(img_neutral, caption="Neutral")
                        log_dict["visual/cond_Yau"] = wandb.Image(img_cond, caption=f"{item['type']}\nCLIP-I: {clip_i:.3f}, MSE: {au_mse:.3f}")
                        if current_mask is not None:
                            log_dict["visual/spatial_mask"] = wandb.Image(current_mask, caption=f"Mask for Step {step}")

                    wandb.log(log_dict, step=step) 

            except Exception as e:
                print(f"\n[警告] Step {step}: AU 提取异常，已跳过。原因: {e}")
                
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # ==========================================
    # 打印并记录最终分类指标
    # ==========================================
    print(f"\n==========================================")
    print(f"评估完成! 总有效样本数: {total_valid}/{len(test_cases)}")
    
    if single_valid > 0:
        avg_single_clip = single_clip_i / single_valid
        avg_single_mse = single_au_mse / single_valid
        wandb.log({"final_single/avg_clip_i": avg_single_clip, "final_single/avg_au_mse": avg_single_mse})
        print(f"[单独 AU] 有效样本: {single_valid} -> CLIP-I: {avg_single_clip:.4f} | AU MSE: {avg_single_mse:.4f}")

    if combo_valid > 0:
        avg_combo_clip = combo_clip_i / combo_valid
        avg_combo_mse = combo_au_mse / combo_valid
        wandb.log({"final_combo/avg_clip_i": avg_combo_clip, "final_combo/avg_au_mse": avg_combo_mse})
        print(f"[组合 AU] 有效样本: {combo_valid} -> CLIP-I: {avg_combo_clip:.4f} | AU MSE: {avg_combo_mse:.4f}")

    if total_valid > 0:
        avg_total_clip = total_clip_i / total_valid
        avg_total_mse = total_au_mse / total_valid
        wandb.log({"final_total/avg_clip_i": avg_total_clip, "final_total/avg_au_mse": avg_total_mse})
        print(f"[总体平均] 有效样本: {total_valid} -> CLIP-I: {avg_total_clip:.4f} | AU MSE: {avg_total_mse:.4f}")
        
    print(f"==========================================")
    wandb.finish()

if __name__ == "__main__":
    main()