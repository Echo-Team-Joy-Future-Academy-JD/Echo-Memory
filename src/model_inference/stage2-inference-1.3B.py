import torch
import argparse
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from safetensors.torch import load_file as safe_load_file


def main():
    parser = argparse.ArgumentParser(description="Stage2 inference with action control")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt describing the game scene (e.g., 'A cyberpunk city game scene, a character walking through neon-lit streets')")
    parser.add_argument("--negative_prompt", type=str, default="oversaturated colors, overexposed, static, blurry details, subtitles, artwork, painting, still image, gray tone, worst quality, low quality, JPEG compression artifacts, ugly, deformed, extra fingers, poorly drawn hands, poorly drawn face, malformed, disfigured, deformed limbs, fused fingers, static image, cluttered background, three legs, crowded background, walking backwards", help="Negative prompt to avoid unwanted artifacts")
    parser.add_argument("--action_path", type=str, default=None, help="Path to action JSON file (optional, for non-generalization inference)")
    parser.add_argument("--dit_model_path", type=str, required=True, help="Path to trained DiT model checkpoint")
    parser.add_argument("--text_encoder_path", type=str, required=True, help="Path to text encoder model")
    parser.add_argument("--vae_path", type=str, required=True, help="Path to VAE model")
    parser.add_argument("--output_path", type=str, required=True, help="Output video path")
    parser.add_argument("--height", type=int, default=352, help="Video height")
    parser.add_argument("--width", type=int, default=640, help="Video width")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--tiled", action="store_true", help="Use tiled inference")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--sigma_shift", type=float, default=5.0, help="Sigma shift for scheduler")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="Classifier-free guidance scale")
    
    args = parser.parse_args()
    
    # 加载 pipeline（先加载基础模型）
    print("Loading pipeline...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(path=args.text_encoder_path, offload_device="cpu"),
            ModelConfig(path=args.vae_path, offload_device="cpu"),
        ],
    )
    
    # 加载训练好的 action module checkpoint
    print(f"Loading action module checkpoint from: {args.dit_model_path}")
    checkpoint = safe_load_file(args.dit_model_path)
    
    # 加载 checkpoint 到 dit 模型
    # 注意：checkpoint 可能只包含 action module 的参数（如 action_mlp, self_attn_with_action 等）
    missing_keys, unexpected_keys = pipe.dit.load_state_dict(checkpoint, strict=False)
    if missing_keys:
        print(f"Warning: {len(missing_keys)} keys were missing when loading checkpoint")
        if len(missing_keys) <= 10:
            print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: {len(unexpected_keys)} unexpected keys in checkpoint (will be ignored)")
        if len(unexpected_keys) <= 10:
            print(f"Unexpected keys: {unexpected_keys}")
    
    pipe.enable_vram_management()
    
    # 准备输入参数
    # 根据 GameFactory 论文和原始 pipeline，直接使用 action_path 参数
    # pipeline 会自己处理 action 数据的加载和转换
    pipe_kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "seed": args.seed,
        "tiled": args.tiled,
        "num_inference_steps": args.num_inference_steps,
        "sigma_shift": args.sigma_shift,
        "cfg_scale": args.cfg_scale,
    }
    
    # 如果提供了 action 文件，直接使用 action_path 参数
    # pipeline 会按照原始逻辑处理（使用 pre_pitch/pre_yaw，取前 80 帧）
    if args.action_path is not None:
        print(f"Using action_path: {args.action_path}")
        pipe_kwargs["action_path"] = args.action_path
    
    # 执行推理
    print("Starting inference...")
    video = pipe(**pipe_kwargs)
    
    # 保存视频
    print(f"Saving video to: {args.output_path}")
    save_video(video, args.output_path, fps=15, quality=5)
    print("Inference completed!")


if __name__ == "__main__":
    main()
