import cv2
import os
import torch
from pathlib import Path
from tqdm import tqdm
from facenet_pytorch import MTCNN
from PIL import Image

# 1. 配置路径 (请确保这里指向你刚才解压 DISFA 的正确路径)
# 这里假设你的原始视频存放在 Video_Files 文件夹中
VIDEO_DIR = Path('/root/autodl-tmp/dataset/disfa/DISFA/Video_RightCamera') 
# 按照 fineface 训练脚本 train.sh 的要求，输出目录必须叫 aligned
OUTPUT_DIR = Path('/root/autodl-tmp/dataset/disfa/aligned')

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 自动调用服务器的 GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用的计算设备: {device}")

    # 2. 初始化 MTCNN 人脸检测与对齐模型
    # image_size=512 匹配 fineface 扩散模型需要的分辨率
    # margin=80 让人脸保留一些边缘背景，更有利于表情生成
    mtcnn = MTCNN(
        image_size=512, 
        margin=80, 
        keep_all=False, 
        post_process=False, 
        device=device
    )

    video_paths = list(VIDEO_DIR.glob('*.avi'))
    if not video_paths:
        print(f"未在 {VIDEO_DIR} 找到 .avi 视频，请检查视频路径是否正确！")
        return

    # 3. 遍历处理每个被试者的视频
    for video_path in tqdm(video_paths, desc="处理进度 (Total Subjects)"):
        subject_id = video_path.stem  # 提取类似 "SN001" 的编号
        subject_out_dir = OUTPUT_DIR / subject_id
        subject_out_dir.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with tqdm(total=total_frames, desc=f"抽取视频 {subject_id}", leave=False) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 必须严格按照 5 位数字命名，这是 fineface 代码里写死的规则
                out_file = subject_out_dir / f"{frame_idx:05d}.jpg"

                # 防止中断重跑时浪费时间，存在则跳过
                if not out_file.exists():
                    # 转换色彩通道
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    
                    try:
                        # 检测、对齐人脸并直接保存为图片
                        mtcnn(img, save_path=str(out_file))
                    except Exception:
                        pass
                    
                    # ⚠️ 极其关键的安全机制：
                    # 如果某帧闭眼或侧脸导致 MTCNN 没检测到人脸，必须强行存一张原图的 resize 版本。
                    # 因为 DISFA 的 AU 标签是连续的，少一帧会导致 Dataloader 找不到文件而中断训练。
                    if not out_file.exists():
                        img.resize((512, 512)).save(out_file)

                frame_idx += 1
                pbar.update(1)

        cap.release()
        
    print("✅ DISFA 数据集预处理圆满完成！")

if __name__ == '__main__':
    main()