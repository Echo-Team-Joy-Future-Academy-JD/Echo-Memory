import json
import os
from typing import Optional

import wandb

from src.model_training.transformers_compat import patch_transformers_hybrid_cache

patch_transformers_hybrid_cache()

from diffsynth.trainers.utils import ModelLogger as BaseModelLogger


class ModelLogger(BaseModelLogger):
    """Compatibility wrapper for legacy training scripts."""

    def __init__(
        self,
        output_path,
        remove_prefix_in_ckpt=None,
        state_dict_converter=lambda x: x,
        wandb_run_name=None,
        ckpt_interval=None,
        resume_step_count=0,
        save_full_model=False,
        context_drop_prob: float = 0.0,
        enable_video_sampling=False,
        sampling_interval_steps: int = 0,
        sampling_two_chunk_memory: bool = False,
        sampling_action_path: Optional[str] = None,
        sampling_two_chunk_action_path: Optional[str] = None,
        sampling_negative_prompt: str = "oversaturated colors, overexposed, static, blurry details",
        sampling_height: int = 352,
        sampling_width: int = 640,
        sampling_num_frames: int = 81,
        sampling_num_inference_steps: int = 50,
        context_memory_frames: int = 1,
        context_source: str = "replay",
        context_per_frame_vae: bool = False,
    ):
        super().__init__(output_path, remove_prefix_in_ckpt=remove_prefix_in_ckpt, state_dict_converter=state_dict_converter)
        self.wandb_run_name = wandb_run_name
        self.ckpt_interval = int(ckpt_interval) if ckpt_interval else None
        self.step_count = int(resume_step_count)
        self.save_full_model = bool(save_full_model)
        self.total_steps = None
        self.context_drop_prob = float(context_drop_prob)
        self.enable_video_sampling = bool(enable_video_sampling)
        self.sampling_interval_steps = int(sampling_interval_steps)
        self.sampling_two_chunk_memory = bool(sampling_two_chunk_memory)
        self.sampling_action_path = sampling_action_path
        self.sampling_two_chunk_action_path = sampling_two_chunk_action_path
        self.sampling_negative_prompt = sampling_negative_prompt
        self.sampling_height = int(sampling_height)
        self.sampling_width = int(sampling_width)
        self.sampling_num_frames = int(sampling_num_frames)
        self.sampling_num_inference_steps = int(sampling_num_inference_steps)
        self.context_memory_frames = int(context_memory_frames)
        self.context_source = context_source.strip().lower()
        self.context_per_frame_vae = bool(context_per_frame_vae)
        self.wandb_logger = None
        if self.wandb_run_name:
            self.wandb_logger = wandb.init(project="wan-cam", name=self.wandb_run_name, reinit=True)

    def _save_step_or_epoch_ckpt(self, accelerator, model, path: str):
        state_dict = None
        unwrapped = accelerator.unwrap_model(model)
        if self.save_full_model:
            # Save full DiT (including action/camera/memory modules), not whole pipeline.
            state_dict = accelerator.get_state_dict(unwrapped.pipe.dit)
            for module_name in ("spatial_memory_module", "spatial_memory_readout_module"):
                module = getattr(unwrapped, module_name, None)
                if module is not None:
                    state_dict.update(
                        {
                            f"{module_name}.{name}": param
                            for name, param in accelerator.get_state_dict(module).items()
                        }
                    )
        if state_dict is None:
            full_state = accelerator.get_state_dict(model)
            state_dict = unwrapped.export_trainable_state_dict(full_state, remove_prefix=self.remove_prefix_in_ckpt)
        state_dict = self.state_dict_converter(state_dict)
        os.makedirs(self.output_path, exist_ok=True)
        accelerator.save(state_dict, path, safe_serialization=True)

    def _maybe_sample_paper_process(self, accelerator=None, model=None, current_batch=None):
        if not (
            self.enable_video_sampling
            and self.sampling_two_chunk_memory
            and self.sampling_interval_steps > 0
            and self.step_count % self.sampling_interval_steps == 0
            and accelerator is not None
            and model is not None
            and current_batch is not None
        ):
            return
        from diffsynth import save_video
        from src.model_training.multichunk_sample_utils import (
            run_two_chunk_memory_monitor,
            sync_pipe_memory_from_training_module,
        )

        sample = current_batch[0] if isinstance(current_batch, list) else current_batch
        first_frame = sample["video"][0]
        unwrapped = accelerator.unwrap_model(model)
        pipe = unwrapped.pipe
        sync_pipe_memory_from_training_module(pipe, unwrapped)
        action0 = self.sampling_two_chunk_action_path or self.sampling_action_path
        action1 = self.sampling_action_path
        frames0, frames1, meta = run_two_chunk_memory_monitor(
            pipe,
            prompt=sample.get("prompt") or sample.get("description") or "A scene.",
            negative_prompt=self.sampling_negative_prompt,
            action_path=self.sampling_action_path,
            chunk0_action_path=action0,
            chunk1_action_path=action1,
            first_frame_pil=first_frame,
            context_memory_frames=self.context_memory_frames,
            chunk_frames=self.sampling_num_frames,
            h=self.sampling_height,
            w=self.sampling_width,
            seed=42 + self.step_count + accelerator.process_index,
            sigma_shift=5.0,
            num_inference_steps=self.sampling_num_inference_steps,
            cfg_scale=5.0,
            inference_noise_level=0.0,
            omit_context_actions=False,
            context_source=self.context_source,
            context_position=os.environ.get("CONTEXT_POSITION", "suffix"),
            context_per_frame_vae=self.context_per_frame_vae,
            device=pipe.device,
            log_prefix=f"[paper-sampling][step={self.step_count}]",
        )
        out_dir = os.path.join(self.output_path, "paper_process_sampling")
        os.makedirs(out_dir, exist_ok=True)
        tag = f"step_{self.step_count:07d}_rank{accelerator.process_index}"
        save_video(list(frames0) + list(frames1), os.path.join(out_dir, f"{tag}_pred.mp4"), fps=15, quality=5)
        with open(os.path.join(out_dir, f"{tag}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def on_step_end(self, loss, accelerator=None, model=None, current_batch=None):
        self.step_count += 1
        if self.wandb_logger is not None:
            if accelerator is None or accelerator.is_main_process:
                loss_v = float(loss.detach().float().item())
                self.wandb_logger.log({"train/loss": loss_v, "step": self.step_count})
        if accelerator is not None and accelerator.is_main_process:
            self._maybe_sample_paper_process(accelerator, model, current_batch)
        if accelerator is not None and self.enable_video_sampling and self.sampling_two_chunk_memory and self.sampling_interval_steps > 0:
            accelerator.wait_for_everyone()
        if (
            self.ckpt_interval
            and accelerator is not None
            and model is not None
            and (self.step_count % self.ckpt_interval == 0)
        ):
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                path = os.path.join(self.output_path, f"Step-{self.step_count}.safetensors")
                self._save_step_or_epoch_ckpt(accelerator, model, path)

    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            self._save_step_or_epoch_ckpt(accelerator, model, path)

    def finish(self):
        if self.wandb_logger is not None:
            wandb.finish()
