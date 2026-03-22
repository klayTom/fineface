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
MODEL_WEIGHT_PATH = "/root/autodl-tmp/fineface/fineface/checkpoint-20000"  
NUM_AUS = 12  

# 严格对齐 au_dataset.py 中的 [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
# 注意：Py-Feat 默认的列名是带 0 填充的两位数格式
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
    # 自动下载面部分析权重
    au_detector = Detector(device=device)
    
    return clip_processor, clip_model, au_detector

def get_1650_test_cases():
    """严格按照论文 Testing details 构建 1650 个合成测试用例"""
    prompts = [
        "a photo of a person", "a closeup portrait of a man", "a closeup portrait of a woman",
        "a clear frontal face photo", "a detailed studio headshot", "a cinematic face portrait",
        "a high quality picture of a face", "a realistic portrait of a young boy", "a portrait of an old man",
        "a photo of a young girl", "a professional id photo", "a casual selfie",
        "a beautiful face portrait", "a symmetrical face photo", "a bright portrait photography"
    ]
    
    test_cases = []
    
    for prompt in prompts:
        # 1. 单独 AU 测试: 12种AU * 5种强度 = 60个用例/Prompt
        # 注意！因为训练数据被缩放到了 0-5 分，所以这里的 5 种强度设定为 1 到 5
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
                
        # 2. 组合 AU 测试: 每种 Prompt 随机生成 50 组 AU
        # 随机激活 2-4 个 AU，强度在 1.0 ~ 5.0 之间随机
        for i in range(50):
            au_tensor = torch.zeros((1, NUM_AUS), dtype=torch.float16)
            num_active = np.random.randint(2, 5) 
            active_indices = np.random.choice(NUM_AUS, num_active, replace=False)
            for idx in active_indices:
                au_tensor[0, idx] = np.round(np.random.uniform(1.0, 5.0), 1)
                
            test_cases.append({
                "prompt": prompt,
                "target_au": au_tensor,
                "type": f"combo_random_{i}"
            })
            
    # 共 15 * (60 + 50) = 1650 个用例
    return test_cases

def load_local_fineface(model_path, device="cuda"):
    """绕过原作者写死的下载逻辑，强制加载你本地训练好的权重"""
    print("Initializing SD 2.1 Base Model...")
    # 必须使用你 train.sh 里训练时用的底座模型
    sd_repo_id = "Manojb/stable-diffusion-2-1-base" 
    
    # 1. 加载 UNet 并注入 AU 注意力层
    unet = UNet2DConditionModel.from_pretrained(sd_repo_id, subfolder="unet", torch_dtype=torch.float16)
    unet.set_default_attn_processor()
    hack_unet_attn_layers(unet, AUAttnProcessor)
    
    # 2. 挂载你本地 checkpoint-20000 下的权重
    print(f"Loading custom trained weights from {model_path}...")
    unet.load_state_dict(torch.load(os.path.join(model_path, "attn_processors.ckpt"), map_location=device), strict=False)
    unet.load_attn_procs(model_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="unet")
    
    # 3. 创建 Stable Diffusion Pipeline
    pipe = StableDiffusionPipeline.from_pretrained(sd_repo_id, unet=unet, torch_dtype=torch.float16).to(device)
    
    # 4. 加载你训练好的 AU Encoder
    au_encoder = AUEncoder().to(device)
    au_encoder.load_state_dict(torch.load(os.path.join(model_path, "au_encoder.ckpt"), map_location=device))
    
    # 5. 动态创建一个继承类，注入你的参数
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
    
    # 1. 加载模型
    clip_processor, clip_model, au_detector = init_eval_models(device)
    print("Loading Custom FineFace Pipeline...")
    pipeline = load_local_fineface(MODEL_WEIGHT_PATH, device)
    pipeline.pipe.set_progress_bar_config(disable=True)
    
    test_cases = get_1650_test_cases()
    print(f"Total test cases to generate: {len(test_cases)}")
    
    total_clip_i = 0
    total_au_mse = 0
    valid_samples = 0
    
    os.makedirs("eval_tmp", exist_ok=True)
    fixed_seed = 42 # 极其关键：保证两张图片的人脸身份一致
    
    for step, item in enumerate(tqdm(test_cases, desc="Evaluating")):
        prompt = item["prompt"]
        target_au_tensor = item["target_au"].to(device)
        
       # ---------------------------------------------------------
        # 准备：将 Tensor 转换为 Pipeline 期望的字典格式
        # (严格对应 fineface_pipeline.py 中的 12 个 AU)
        # ---------------------------------------------------------
        PIPELINE_AU_KEYS = ["AU1", "AU2", "AU4", "AU5", "AU6", "AU9", "AU12", "AU15", "AU17", "AU20", "AU25", "AU26"]
        
        target_au_dict = {}
        for idx, key in enumerate(PIPELINE_AU_KEYS):
            val = target_au_tensor[0, idx].item()
            if val > 0:
                target_au_dict[key] = val

        # ---------------------------------------------------------
        # A. 生成无表情基准图 (Y_0)
        # ---------------------------------------------------------
        generator_neutral = torch.Generator(device=device).manual_seed(fixed_seed)
        # 传入 aus={}，空字典意味着所有 AU 强度均为 0
        img_neutral = pipeline(prompt=prompt, aus={}, generator=generator_neutral).images[0]
        
        # ---------------------------------------------------------
        # B. 生成带表情目标图 (Y_au)
        # ---------------------------------------------------------
        generator_cond = torch.Generator(device=device).manual_seed(fixed_seed)
        # 传入 aus=target_au_dict，准确赋予各个 AU 目标强度
        img_cond = pipeline(prompt=prompt, aus=target_au_dict, generator=generator_cond).images[0]
        
        # ---------------------------------------------------------
        # C. 计算 CLIP-I (身份保持度)
        # ---------------------------------------------------------
        inputs_neutral = clip_processor(images=img_neutral, return_tensors="pt").to(device)
        inputs_cond = clip_processor(images=img_cond, return_tensors="pt").to(device)
        with torch.no_grad():
            emb_neutral = clip_model(**inputs_neutral).image_embeds
            emb_cond = clip_model(**inputs_cond).image_embeds
        clip_i = F.cosine_similarity(emb_neutral, emb_cond).item()
        
        # ---------------------------------------------------------
        # D. 计算 AU MSE (表情精准度)
        # ---------------------------------------------------------
        tmp_path = f"eval_tmp/cond_{step}.jpg"
        img_cond.save(tmp_path)
        
        # ---------------------------------------------------------
        # D. 计算 AU MSE (表情精准度)
        # ---------------------------------------------------------
        tmp_path = f"eval_tmp/cond_{step}.jpg"
        img_cond.save(tmp_path)
        
        prediction = au_detector.detect_image(tmp_path)
        if len(prediction) > 0:
            try:
                # 1. 取出结果并尝试强制转换为 float32
                # 如果没检测到脸，这里可能会变成 NaN 或者抛出异常
                pred_aus = prediction[TARGET_AU_NAMES].iloc[0].values.astype(np.float32)
                
                # 2. 安全检查：如果数组里有 NaN（空值），说明人脸崩了，直接跳过
                if np.isnan(pred_aus).any():
                    print(f"\n[提示] Step {step}: 未检测到有效人脸(可能生成了扭曲的图像)，跳过指标计算。")
                else:
                    # 3. 正常计算 MSE
                    pred_au_tensor = torch.tensor(pred_aus, dtype=torch.float32) * 5.0
                    target_au_cpu = target_au_tensor.squeeze(0).float().cpu()
                    
                    au_mse = F.mse_loss(pred_au_tensor, target_au_cpu).item()
                    
                    total_clip_i += clip_i
                    total_au_mse += au_mse
                    valid_samples += 1
                    
                    # 4. 记录到 Wandb
                    if step % 50 == 0:
                        wandb.log({
                            "eval/step_clip_i": clip_i,
                            "eval/step_au_mse": au_mse,
                            "visual/neutral_Y0": wandb.Image(img_neutral, caption=f"Neutral"),
                            "visual/cond_Yau": wandb.Image(img_cond, caption=f"{item['type']}\nCLIP-I: {clip_i:.3f}, MSE: {au_mse:.3f}")
                        }, step=step)
                    else:
                        wandb.log({
                            "eval/step_clip_i": clip_i,
                            "eval/step_au_mse": au_mse,
                        }, step=step)
                        
            except Exception as e:
                # 如果数据格式严重错误，捕获异常并跳过，防止脚本崩溃
                print(f"\n[警告] Step {step}: AU 提取异常，已跳过。原因: {e}")
                
        # 删除临时文件节省空间
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # ---------------------------------------------------------
    # E. 打印并记录最终指标
    # ---------------------------------------------------------
    if valid_samples > 0:
        avg_clip_i = total_clip_i / valid_samples
        avg_au_mse = total_au_mse / valid_samples
        wandb.log({
            "final/avg_clip_i": avg_clip_i,
            "final/avg_au_mse": avg_au_mse
        })
        print(f"\n==========================================")
        print(f"评估完成! 有效样本数: {valid_samples}/{len(test_cases)}")
        print(f"论文对齐指标 -> 平均 CLIP-I: {avg_clip_i:.4f} (越高越好，接近 1)")
        print(f"论文对齐指标 -> 平均 AU MSE: {avg_au_mse:.4f} (越低越好)")
        print(f"==========================================")
        
    wandb.finish()

if __name__ == "__main__":
    main()