# Memory Baselines Basic: FAR/FramePack + Spatial + VideoSSM

本目录添加三组「基础记忆模块」消融 baseline（与 `memory_design_advanced/` 的“先进检索/统一隐式”正交）：

## 论文 / 代码对应（命名）

| 论文叙述（建议） | 行为 | 代码 |
|------------------|------|------|
| **FramePack-Weight**（FAR 风格） | 仅对 context patch tokens 时间衰减 + 全局 scale，**不改变 K** | `diffsynth/models/memory/framepack_weight.py` → `apply_framepack_token_weights`；`model_fn_wan_video` 内调用 |
| **FramePack-Length** | latent 时间维 r 组 mean + 与 RT 同规则 pad/mean | `diffsynth/models/memory/framepack_length.py`（`framepack_length_compress_context_latents` / `framepack_align_context_actions_to_latents`）；`wan_video_new` 训练与推理路径调用 |
| **Spatial（grid memory）** | time-mean → G×G → M tokens | `diffsynth/models/memory/spatial_grid_memory.py`（`SpatialGridMemory`）；`inject_spatial_memory(..., mode)`：`concat_text`（默认）、`none`、`cross_attn_readout`（真实 readout：target Q 读 mem K/V，zero-init gate） |
| **块内 SSM（论文向）** | DiT 内 block-wise 时序 SSM | `--use_block_wise_ssm`（arXiv:2505.20171 叙事） |
| **VideoSSM hybrid（legacy）** | 深度可分卷积式 lite 基线 | `diffsynth/models/memory/videossm_hybrid.py`（`HybridStateSpaceMemory`），`--use_videossm_hybrid`；与 block-wise SSM **二选一**做叙事更清晰 |

**训练监控（2-chunk）**：`--enable_video_sampling --sampling_two_chunk_memory`。当前实现固定 `chunk0=left_45`、`chunk1=right_45`（从 `sampling_two_chunk_action_path` 所在目录解析 `action_rotation_left_45.json`/`action_rotation_right_45.json`）。与 `--sampling_atomic_left_right` / `--sampling_four_prompts` / `--sampling_two_prompts` **按优先级互斥**（见 `ModelLogger.on_step_end` 注释）。输出：`sampling_videos/step_*_two_chunk_memory*.mp4` 与 `step_*_two_chunk_meta.json`。两段之间 **无跨 chunk 的 SSM/RNN 隐状态传递**（与当前 diffusion 采样一致）。

### 2-chunk 消融专用脚本（新建）

共用片段：先 `source common_env.sh`，再 `source common_sampling_two_chunk.sh`（定义 `SAMPLING_TWO_CHUNK_FLAGS`，含 left/right_45 动作路径与 `--use_anchor_frame` 默认开启）。

| 脚本 | 说明 |
|------|------|
| `run_ablation_no_memory_baseline_two_chunk.sh` | 对照：无 FramePack/Spatial/SSM 额外开关，仅 replay context + 2-chunk 监控 |
| `run_ablation_framepack_weight_two_chunk.sh` | FramePack-Weight + FOV（与 `run_framepack_baseline.sh` 一致）+ 2-chunk |
| `run_ablation_framepack_len_r2_two_chunk.sh` | FramePack-Length r=2 + replay + 2-chunk |
| `run_ablation_framepack_len_r4_two_chunk.sh` | FramePack-Length r=4 + replay + 2-chunk |
| `run_ablation_spatial_concat_text_two_chunk.sh` | Spatial + `--spatial_memory_inject_mode concat_text` + 2-chunk |
| `run_ablation_spatial_inject_none_two_chunk.sh` | Spatial + `inject_mode none`（与 concat 对照） |
| `run_ablation_spatial_cross_attn_readout_two_chunk.sh` | `cross_attn_readout`（真实 readout 分支） |
| `run_ablation_block_wise_ssm_two_chunk.sh` | `--use_block_wise_ssm`（论文向）+ replay + 2-chunk |
| `run_ablation_videossm_hybrid_two_chunk.sh` | legacy `--use_videossm_hybrid` + FOV + 2-chunk |
| `run_all_ablations_two_chunk.sh` | 按顺序执行以上脚本（耗时极长；请按需注释其中行） |

原有 `run_framepack_baseline.sh` 等仍默认 **`--sampling_atomic_left_right`**，与 2-chunk **互斥**；要做 2-chunk 消融请用上表脚本或自行改参。

## 1) FAR / FramePack-style（压缩式记忆：重权重）

- 思路：不改变 Context token 序列长度（避免破坏 Context-as-Memory 的训练/推理对齐），而是在 `model_fn_wan_video` 中对 **context tokens 做 per-frame decay + 全局缩放**。
- 开关：
  - `--use_framepack_memory`
  - `--context_temporal_decay <float>`（默认 1.0）
  - `--context_attention_weight <float>`（默认 1.0）

脚本：`run_framepack_baseline.sh`

### FramePack **长度**压缩（`run_framepack_lencompress_r2.sh` / `r4.sh`）

- 对 `context_latents` 时间维做 **不重叠、每组 r 帧 mean pool**（`K → K'`）。若 `K % r != 0`，对 latent **末尾帧复制**（replicate-last）padding 到 `r` 的倍数，再分组；与推理 `__call__` **同一套函数**（`framepack_length_compress_context_latents`）。
- **RT `context_actions`**：先对齐到 `K` 行，再与 latent 做 **相同** 的 padding，再按 **相同 r 组** mean，与训练 `training_loss` 一致（`framepack_align_context_actions_to_latents`），避免只压 latent、不压 pose 的错位。
- 本目录与 `run_spatial_memory_baseline.sh` 三脚本已统一：`--context_source replay --prev_chunk_frames 81`，与 `run_replay_loop_two_chunk` 的 context 构造一致；`--context_memory_frames 5` 与续段 `ctx=5` 一致。

## 2) Spatial Memory baseline（长时空间记忆：追加 cross-attn tokens）

- **默认（推荐）**：可学习的 **SpatialGridMemory**（`diffsynth/models/memory/spatial_grid_memory.py`，旧路径 `diffsynth/models/spatial_grid_memory.py` 仍 re-export）：time-mean → `G×G` 自适应空间池化 → 可学习矩阵混合到 `M` 个 token → 经 `inject_spatial_memory` 拼到 text cross-attn（`--spatial_memory_inject_mode concat_text`，默认）。
- **旧版（仅复现）**：`--use_spatial_memory_legacy`：非可学习的 time-mean + `adaptive_avg_pool1d`。
- 开关：
  - `--use_spatial_memory`
  - `--use_spatial_memory_legacy`（旧池化 baseline）
  - `--spatial_memory_tokens <int>`（默认 64）
  - `--spatial_memory_grid <int>`（默认 8，仅非 legacy）

脚本：`run_spatial_memory_baseline.sh`

## Context 构造与多 chunk 对齐（训练）

- `--context_source fov`：FOV/overlap 检索（CAM 论文线）；**本目录 baseline 脚本已改为不用 fov**。
- `--context_source replay` + `--prev_chunk_frames 81`：与 `run_replay_loop_two_chunk` 相同的 **虚拟 chunk1 + `context_frames_for_next_chunk`**（`context_chunk_utils.py`）。
- `--context_source prev_chunk_tail`：从磁盘加载 **`[start_frame-N, start_frame)`** 连续 N 帧（需 `dataset_base_path/frames/...`）。

共享实现：`src/model_training/context_chunk_utils.py`。

## 3) VideoSSM baseline（Hybrid State-Space Memory，**legacy**）

- 思路：在 `DiTBlock_w_Action` 内新增 `HybridStateSpaceMemory`（`diffsynth/models/memory/videossm_hybrid.py`，简化版）：对 token 按 (T latent frames)×(spatial tokens) 组织，对每个 spatial 位置沿时间做 causal depthwise conv（SSM-like），并用 gate 控制强度（0-init → 初期≈identity）。**论文向长时序 SSM 叙事请优先 `--use_block_wise_ssm`**，与本 baseline 区分。
- 开关：
  - `--use_videossm_hybrid`
  - `--videossm_every_n_blocks <int>`（默认 4）
  - `--videossm_kernel_size <int>`（默认 3）
  - `--videossm_expand <int>`（默认 2）

脚本：`run_videossm_hybrid_baseline.sh`

## 推理评测对齐

训练 `common_env.sh` 中的 `CONTEXT_POSITION`、`USE_CONCATENATION_INFERENCE`、`USE_RT_RELATIVE` 会影响 `WanVideoPipeline` 行为。评测时请使用：

- `ab_study/exp1_4_4_cam_rt_paper_style/eval_infer_alignment_env.sh`（已被 `run_evals_ep0_pre_qkv_rt_merge.sh`、`eval_v2/run_*.sh` 自动 source）
- 多 chunk + `context_memory_frames=5` 时，`run_one_chunk` 在 `ctx>1` 时会传入 `context_actions`（与训练一致），而非 omit。

### 压缩记忆 / 空间记忆 与 `memory_baseline_runtime.py`

| 本目录训练脚本 | 推理侧（ckpt 路径匹配） |
|----------------|-------------------------|
| `run_framepack_baseline.sh`：`--use_framepack_memory --context_temporal_decay 0.9 --context_attention_weight 1.0` | `use_framepack_memory` + 同上；**不**开 `use_framepack_length_compress` |
| `run_framepack_lencompress_r2.sh` / `r4.sh`：`--use_framepack_length_compress --framepack_ratio 2/4` | `use_framepack_length_compress` + `framepack_ratio`；**不**开 `use_framepack_memory` |
| `run_spatial_memory_baseline.sh`：`--use_spatial_memory --spatial_memory_tokens 64`（默认 **SpatialGridMemory** grid=8，非 `--use_spatial_memory_legacy`） | `use_spatial_memory` + tokens=64 + legacy=False；**`loop_utils.load_pipeline_and_ckpt` 从 ckpt 加载 `spatial_memory_module.*`**；若无权重会警告并降级 legacy pool（与训练不等价） |

实现与对照表见：`../memory_baseline_runtime.py` 文件头文档字符串。
