#!/usr/bin/env python3
"""
Long Video Generation Script
Generates 30-second level videos using iterative context-based generation:
1. First generates 81 frames (reference: Experiment 17)
2. Uses the last K frames as context for next generation
3. Repeats until reaching 30-second level video
4. Prompts are randomly sampled from training dataset
"""

import os
import sys
import json
import argparse
import random
import torch
from PIL import Image
from tqdm import tqdm

# Add project root to path
current_file_abs = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_abs)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import VideoDataset


def load_model(checkpoint_path, model_paths, lora_path=None, lora_alpha=1.0, device="cuda"):
    """Load model from checkpoint"""
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Load base pipeline
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
        ],
    )
    
    # Load LoRA if specified
    if lora_path and os.path.exists(lora_path):
        print(f"Loading LoRA from: {lora_path}")
        pipe.load_lora(pipe.dit, lora_path, alpha=lora_alpha)
    
    # Load checkpoint if specified
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        from safetensors.torch import load_file as safe_load_file
        checkpoint = safe_load_file(checkpoint_path)
        pipe.dit.load_state_dict(checkpoint, strict=False)
    
    pipe.enable_vram_management()
    pipe.eval()
    
    return pipe


def sample_prompts_from_dataset(dataset, num_prompts=5):
    """Randomly sample prompts from dataset"""
    prompts = []
    dataset_size = len(dataset)
    
    if dataset_size == 0:
        print("Warning: Dataset is empty, using default prompts")
        return ["A cyberpunk city game scene, a character walking through neon-lit streets"] * num_prompts
    
    # Sample random indices
    indices = random.sample(range(dataset_size), min(num_prompts, dataset_size))
    
    print(f"Sampling {len(indices)} prompts from dataset (size: {dataset_size})...")
    for idx in indices:
        try:
            sample = dataset[idx]
            if isinstance(sample, dict):
                prompt = sample.get("description") or sample.get("prompt") or sample.get("text", "")
                if prompt:
                    prompts.append(prompt)
                else:
                    print(f"Warning: Sample {idx} has no prompt field, skipping")
            else:
                print(f"Warning: Sample {idx} is not a dict, skipping")
        except Exception as e:
            print(f"Warning: Failed to load sample {idx}: {e}, skipping")
    
    # Fill with default if not enough prompts
    while len(prompts) < num_prompts:
        prompts.append("A cyberpunk city game scene, a character walking through neon-lit streets")
    
    return prompts[:num_prompts]


def encode_frames_to_latents(pipe, frames):
    """Encode frames to latents using VAE"""
    pipe.load_models_to_device(["vae"])
    vae = pipe.vae
    
    latents_list = []
    for frame in frames:
        vid = pipe.preprocess_video([frame]).squeeze(0)
        with torch.no_grad():
            lat = vae.encode([vid], device=pipe.device)[0].unsqueeze(0)
            latents_list.append(lat)
    
    if latents_list:
        return torch.cat(latents_list, dim=2)
    return None


def generate_long_video(
    pipe,
    prompt,
    negative_prompt="oversaturated colors, overexposed, static, blurry details",
    output_dir="./long_video_output",
    video_name="long_video",
    context_memory_frames=4,
    frames_per_segment=81,
    target_frames=450,  # 30 seconds at 15fps
    height=352,
    width=640,
    num_inference_steps=20,
    cfg_scale=5.0,
    timestep_shift=1.0,
    seed=42,
    fps=15,
):
    """
    Generate long video using iterative context-based generation
    
    Args:
        pipe: WanVideoPipeline instance
        prompt: Text prompt for generation
        negative_prompt: Negative prompt
        output_dir: Output directory for videos
        video_name: Base name for output video
        context_memory_frames: Number of context frames to use (K)
        frames_per_segment: Frames to generate per segment (default: 81)
        target_frames: Target total frames (default: 450 for 30s at 15fps)
        height: Video height
        width: Video width
        num_inference_steps: Number of inference steps
        cfg_scale: CFG scale
        timestep_shift: Timestep shift
        seed: Random seed
        fps: FPS for output video
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set environment variable for concatenation inference
    os.environ["USE_CONCATENATION_INFERENCE"] = "true"
    
    all_frames = []
    current_context_latents = None
    current_context_frames = []
    
    # Calculate number of segments needed
    num_segments = (target_frames + frames_per_segment - 1) // frames_per_segment
    
    print(f"Generating long video: {target_frames} frames in {num_segments} segments")
    print(f"  - Frames per segment: {frames_per_segment}")
    print(f"  - Context frames: {context_memory_frames}")
    print(f"  - Prompt: {prompt[:100]}...")
    
    torch.manual_seed(seed)
    
    for segment_idx in range(num_segments):
        # Calculate frames to generate for this segment
        remaining_frames = target_frames - len(all_frames)
        frames_to_generate = min(frames_per_segment, remaining_frames)
        
        if frames_to_generate <= 0:
            break
        
        print(f"\n[{segment_idx + 1}/{num_segments}] Generating {frames_to_generate} frames...")
        
        # Prepare sampling kwargs
        # First segment: no context (generate from scratch)
        # Subsequent segments: use context from previous segment
        has_context = current_context_latents is not None and segment_idx > 0
        
        sampling_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_frames": frames_to_generate,
            "num_inference_steps": num_inference_steps,
            "seed": seed + segment_idx,  # Different seed for each segment
            "cfg_scale": cfg_scale,
            "sigma_shift": timestep_shift,
            "denoising_strength": 1.0,
        }
        
        # Add context memory only if we have context
        if has_context:
            sampling_kwargs["enable_context_memory"] = True
            sampling_kwargs["context_latents"] = current_context_latents
            sampling_kwargs["num_context_frames"] = len(current_context_frames)
        
        try:
            # Generate frames
            if has_context:
                print(f"  Using {len(current_context_frames)} context frames from previous segment...")
            
            generated_frames = pipe(**sampling_kwargs)
            
            if isinstance(generated_frames, list):
                segment_frames = generated_frames
            else:
                segment_frames = [generated_frames] if hasattr(generated_frames, '__iter__') else [generated_frames]
            
            # Add to all frames
            all_frames.extend(segment_frames)
            
            # Update context: use last K frames from generated segment
            # These will be used as context for the next segment
            if len(segment_frames) >= context_memory_frames:
                context_frames = segment_frames[-context_memory_frames:]
                current_context_frames = context_frames
                
                # Encode context frames to latents
                print(f"  Encoding last {context_memory_frames} frames as context for next segment...")
                current_context_latents = encode_frames_to_latents(pipe, context_frames)
            else:
                # If not enough frames, use all frames as context
                current_context_frames = segment_frames
                current_context_latents = encode_frames_to_latents(pipe, segment_frames)
            
            print(f"  Generated {len(segment_frames)} frames (total: {len(all_frames)}/{target_frames})")
            
        except Exception as e:
            print(f"  Error generating segment {segment_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Save final video
    if len(all_frames) > 0:
        output_path = os.path.join(output_dir, f"{video_name}.mp4")
        print(f"\nSaving video to: {output_path}")
        print(f"  Total frames: {len(all_frames)}")
        print(f"  Duration: {len(all_frames) / fps:.2f} seconds")
        
        save_video(all_frames, output_path, fps=fps, quality=5)
        print(f"Video saved: {output_path}")
        
        # Save prompt
        prompt_path = os.path.join(output_dir, f"{video_name}_prompt.txt")
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        return output_path
    else:
        print("Error: No frames generated")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate long videos using iterative context-based generation")
    
    # Model paths
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA weights")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA alpha")
    parser.add_argument("--model_paths", type=str, default=None, help="JSON string of model paths (not used if checkpoint_path is set)")
    
    # Dataset
    parser.add_argument("--dataset_base_path", type=str, required=True, help="Base path to dataset")
    parser.add_argument("--dataset_metadata_path", type=str, required=True, help="Path to dataset metadata CSV")
    parser.add_argument("--num_prompts", type=int, default=5, help="Number of prompts to sample from dataset")
    
    # Generation parameters
    parser.add_argument("--output_dir", type=str, default="./long_video_output", help="Output directory")
    parser.add_argument("--context_memory_frames", type=int, default=4, help="Number of context frames (K)")
    parser.add_argument("--frames_per_segment", type=int, default=81, help="Frames per segment (default: 81)")
    parser.add_argument("--target_frames", type=int, default=450, help="Target total frames (30s at 15fps)")
    parser.add_argument("--height", type=int, default=352, help="Video height")
    parser.add_argument("--width", type=int, default=640, help="Video width")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="CFG scale")
    parser.add_argument("--timestep_shift", type=float, default=1.0, help="Timestep shift")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fps", type=int, default=15, help="FPS for output video")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Load dataset for prompt sampling
    print("Loading dataset...")
    from diffsynth.trainers.utils import wan_parser
    dataset_args = wan_parser.parse_args([])  # Create minimal args
    dataset_args.dataset_base_path = args.dataset_base_path
    dataset_args.dataset_metadata_path = args.dataset_metadata_path
    dataset_args.height = args.height
    dataset_args.width = args.width
    
    dataset = VideoDataset(args=dataset_args)
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Sample prompts
    prompts = sample_prompts_from_dataset(dataset, args.num_prompts)
    print(f"Sampled {len(prompts)} prompts")
    
    # Load model
    model_paths = None
    if args.model_paths:
        model_paths = json.loads(args.model_paths)
    
    pipe = load_model(
        checkpoint_path=args.checkpoint_path,
        model_paths=model_paths,
        lora_path=args.lora_path,
        lora_alpha=args.lora_alpha,
        device=args.device,
    )
    
    # Generate videos for each prompt
    output_paths = []
    for idx, prompt in enumerate(prompts):
        print(f"\n{'='*80}")
        print(f"Generating video {idx + 1}/{len(prompts)}")
        print(f"{'='*80}")
        
        video_name = f"long_video_{idx + 1:03d}"
        
        output_path = generate_long_video(
            pipe=pipe,
            prompt=prompt,
            output_dir=args.output_dir,
            video_name=video_name,
            context_memory_frames=args.context_memory_frames,
            frames_per_segment=args.frames_per_segment,
            target_frames=args.target_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            timestep_shift=args.timestep_shift,
            seed=args.seed + idx,  # Different seed for each video
            fps=args.fps,
        )
        
        if output_path:
            output_paths.append(output_path)
    
    print(f"\n{'='*80}")
    print(f"Generation completed: {len(output_paths)} videos generated")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

