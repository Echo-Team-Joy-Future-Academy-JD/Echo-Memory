"""
Non-Generalization Inference for GameFactory
============================================

This script implements non-generalization inference as described in the GameFactory paper:
"GameFactory: Creating New Games with Generative Interactive Videos" (https://arxiv.org/abs/2501.08325)

Non-generalization inference:
- Uses fixed action sequences from training data
- Tests model performance on known scenes and action sequences
- Validates action control module accuracy on training distribution

Key features:
- Loads action sequences from JSON files (same format as training data)
- Uses pre-trained action control module
- Generates videos with precise action control
"""

import torch
import argparse
import os
import sys
import importlib
import logging
from datetime import datetime

# CRITICAL: Ensure we use local code, not installed package
# Add project root to path first to prioritize local code
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Force reload to avoid cached bytecode issues
# Clear modules to force fresh import from local files
modules_to_clear = [
    'diffsynth.pipelines.wan_video_new',
    'diffsynth.pipelines',
    'diffsynth',
]
for mod in modules_to_clear:
    if mod in sys.modules:
        del sys.modules[mod]

# Import from wan_video_new explicitly (not from __init__.py which imports wan_video.py)
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from safetensors.torch import load_file as safe_load_file

# Setup logging
def setup_logging(log_file=None):
    """Setup logging to both file and console"""
    if log_file is None:
        log_dir = os.path.join(project_root, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"non_gen_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Create logger
    logger = logging.getLogger('non_gen_inference')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (also output to console)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger, log_file

# Initialize logger (will be set up in main)
logger = None


def validate_action_file(action_path, logger):
    """Validate that the action file exists and has the correct format."""
    if not os.path.exists(action_path):
        logger.error(f"Action file not found: {action_path}")
        raise FileNotFoundError(f"Action file not found: {action_path}")
    
    import json
    with open(action_path, 'r') as f:
        data = json.load(f)
    
    if 'actions' not in data:
        logger.error(f"Action file missing 'actions' key: {action_path}")
        raise ValueError(f"Action file missing 'actions' key: {action_path}")
    
    num_actions = len(data['actions'])
    logger.info(f"Action file contains {num_actions} frames")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Non-Generalization Inference for GameFactory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (from GameFactory paper: https://arxiv.org/abs/2501.08325):
  # Single action sequence inference with fixed action sequences from training data
  python stage2-inference-non-generalization.py \\
    --action_path data/actions/seed_1_part_1.json \\
    --prompt "Minecraft game scene, a character exploring in a forest, sunlight filtering through tree leaves" \\
    --dit_model_path checkpoints/stage2-key-mouse-merge/epoch-1.safetensors \\
    --output_path outputs/non_gen_seed_1.mp4

  # Batch inference on multiple action sequences (tests model on known scenes)
  for action_file in data/actions/*.json; do
    python stage2-inference-non-generalization.py \\
      --action_path "$action_file" \\
      --prompt "Minecraft game scene, a character exploring in a forest, sunlight filtering through tree leaves" \\
      --dit_model_path checkpoints/stage2-key-mouse-merge/epoch-1.safetensors \\
      --output_path "outputs/non_gen_$(basename $action_file .json).mp4"
  done
        """
    )
    
    # Required arguments
    parser.add_argument("--action_path", type=str, required=True,
                       help="Path to action JSON file from training data")
    parser.add_argument("--dit_model_path", type=str, required=True,
                       help="Path to trained DiT model checkpoint (action control module)")
    parser.add_argument("--text_encoder_path", type=str, required=True,
                       help="Path to text encoder model")
    parser.add_argument("--vae_path", type=str, required=True,
                       help="Path to VAE model")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output video path")
    
    # Prompt arguments
    parser.add_argument("--prompt", type=str, required=True,
                       help="Text prompt describing the scene (e.g., 'Minecraft game scene, a character exploring in a forest')")
    parser.add_argument("--negative_prompt", type=str,
                       default="oversaturated colors, overexposed, static, blurry details, subtitles, artwork, painting, still image, gray tone, worst quality, low quality, JPEG compression artifacts, ugly, deformed, extra fingers, poorly drawn hands, poorly drawn face, malformed, disfigured, deformed limbs, fused fingers, static image, cluttered background, three legs, crowded background, walking backwards",
                       help="Negative prompt to avoid unwanted artifacts")
    
    # Video parameters (should match training settings)
    parser.add_argument("--height", type=int, default=352,
                       help="Video height (must match training)")
    parser.add_argument("--width", type=int, default=640,
                       help="Video width (must match training)")
    parser.add_argument("--num_frames", type=int, default=81,
                       help="Number of frames (must match training)")
    
    # Inference parameters
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed for reproducibility")
    parser.add_argument("--tiled", action="store_true",
                       help="Use tiled inference for memory efficiency")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Number of diffusion inference steps")
    parser.add_argument("--sigma_shift", type=float, default=5.0,
                       help="Sigma shift for scheduler")
    parser.add_argument("--cfg_scale", type=float, default=5.0,
                       help="Classifier-free guidance scale")
    parser.add_argument("--log_file", type=str, default=None,
                       help="Path to log file (default: auto-generated in logs/ directory)")
    
    args = parser.parse_args()
    
    # Setup logging
    global logger
    logger, log_file = setup_logging(args.log_file)
    logger.info("=" * 60)
    logger.info("GameFactory Non-Generalization Inference")
    logger.info("=" * 60)
    logger.info(f"Paper: https://arxiv.org/abs/2501.08325")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Action file: {args.action_path}")
    
    # Verify we're using the local file (not an installed package)
    try:
        pipeline_module_file = sys.modules['diffsynth.pipelines.wan_video_new'].__file__
        if 'GameFactory-Wan' not in pipeline_module_file:
            logger.warning(f"Pipeline file is not in GameFactory-Wan directory!")
            logger.warning(f"  File: {pipeline_module_file}")
            logger.warning(f"  This may be an installed package version without action_path support!")
    except:
        pass
    
    # Validate action file
    validate_action_file(args.action_path, logger)
    
    # Load pipeline
    logger.info("")
    logger.info("[1/4] Loading pipeline and base models...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(
                model_id="Wan-AI/Wan2.1-T2V-1.3B",
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
                offload_device="cpu"
            ),
            ModelConfig(path=args.text_encoder_path, offload_device="cpu"),
            ModelConfig(path=args.vae_path, offload_device="cpu"),
        ],
    )
    
    # Load trained action control module
    logger.info("")
    logger.info(f"[2/4] Loading action control module from: {args.dit_model_path}")
    checkpoint = safe_load_file(args.dit_model_path)
    
    missing_keys, unexpected_keys = pipe.dit.load_state_dict(checkpoint, strict=False)
    if missing_keys:
        logger.warning(f"  Warning: {len(missing_keys)} keys missing (expected for action-only checkpoints)")
        if len(missing_keys) <= 10:
            logger.debug(f"  Missing: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"  Warning: {len(unexpected_keys)} unexpected keys (will be ignored)")
        if len(unexpected_keys) <= 10:
            logger.debug(f"  Unexpected: {unexpected_keys}")
    
    # Enable VRAM management for memory efficiency
    logger.info("")
    logger.info("[3/4] Enabling VRAM management...")
    pipe.enable_vram_management()
    
    # Prepare inference parameters
    logger.info("")
    logger.info("[4/4] Preparing inference parameters...")
    logger.info(f"  Prompt: {args.prompt}")
    logger.info(f"  Resolution: {args.width}x{args.height}")
    logger.info(f"  Frames: {args.num_frames}")
    logger.info(f"  Inference steps: {args.num_inference_steps}")
    logger.info(f"  CFG scale: {args.cfg_scale}")
    logger.info(f"  Seed: {args.seed}")
    
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
    
    # Add action_path only if provided (required for non-generalization inference)
    if args.action_path is not None:
        pipe_kwargs["action_path"] = args.action_path
    
    # Run inference
    logger.info("")
    logger.info("=" * 60)
    logger.info("Starting inference with fixed action sequence...")
    logger.info("=" * 60)
    
    # Verify we're using the correct pipeline module
    pipeline_module = type(pipe).__module__
    if 'wan_video_new' not in pipeline_module:
        logger.error(f"Wrong pipeline module! Expected 'wan_video_new', got '{pipeline_module}'")
        raise ValueError(
            f"Wrong pipeline module! Expected 'wan_video_new', got '{pipeline_module}'. "
            f"Make sure you're importing from diffsynth.pipelines.wan_video_new"
        )
    
    logger.info(f"✓ Using pipeline from: {pipeline_module}")
    logger.info(f"✓ Action file: {args.action_path}")
    logger.info("")
    
    # Verify the source file has action_path
    import inspect
    try:
        source_file = inspect.getfile(pipe.__call__)
        logger.debug(f"Pipeline __call__ source file: {source_file}")
        
        # Read the actual source code to verify
        with open(source_file, 'r') as f:
            source_lines = f.readlines()
            # Check around line 544 (where action_path should be)
            for i, line in enumerate(source_lines[540:550], start=541):
                if 'action_path' in line:
                    logger.debug(f"  ✓ Found 'action_path' in source at line {i}: {line.strip()}")
                    break
            else:
                logger.warning(f"  ✗ 'action_path' NOT found in source file!")
    except Exception as e:
        logger.debug(f"Could not check source file: {e}")
    
    # Try to get signature (may fail due to @torch.no_grad decorator)
    try:
        unwrapped = getattr(pipe.__call__, '__wrapped__', pipe.__call__)
        sig = inspect.signature(unwrapped)
        params = list(sig.parameters.keys())
        has_action_path = 'action_path' in params
        logger.debug(f"Signature check - Has action_path: {has_action_path}")
        if not has_action_path:
            logger.debug(f"  Last 10 parameters: {params[-10:]}")
    except Exception as e:
        logger.debug(f"Could not inspect signature (this is OK): {e}")
    
    # Directly call the pipeline
    # The action_path parameter exists in wan_video_new.py line 544
    # Even if inspect can't see it, Python should accept it at runtime
    logger.info("Calling pipeline with action_path...")
    video = pipe(**pipe_kwargs)
    
    # Save video
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    logger.info("")
    logger.info(f"Saving video to: {args.output_path}")
    save_video(video, args.output_path, fps=15, quality=5)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Inference completed successfully!")
    logger.info(f"Output: {args.output_path}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

