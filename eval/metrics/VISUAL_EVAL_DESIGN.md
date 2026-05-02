# 可视化评测设计（以肉眼查看为主）

不依赖 PSNR/SSIM 等数值 metric，通过**固定 prompt 与首 chunk（首帧）**，在相同条件下生成视频，便于人工对比不同模型或设置下的表现。

## 1. 设计思路

- **Prompt**：按“考察维度”设计多组文案（身份保持、长时一致性、回环重访、状态/物体等），与首帧语义匹配时生成更稳定，也便于你针对某一维度选看。
- **首 chunk / 首帧**：固定若干“标准首帧”（固定图片或从数据集指定 `(video_name, start_frame)`），同一首帧 + 同一 prompt 下对比不同 ckpt，控制变量。
- **输出组织**：按 `prompt_id` 与 `first_chunk_id` 分目录（如 `evals_visual/prompt_identity_single_person_first_fixed_face/`），每个目录内 2chunk/4chunk 的 MP4，方便按文件夹打开对比。

## 2. 配置文件：prompt 与首帧预设

见 **`visual_eval_config.yaml`**：

- **prompts**：`id`、`text`、`category`（identity / long_horizon / state / loop_closure / generic）、`note`。按需增删或改文案。
- **first_chunk_presets**：
  - `fixed_image`：`path` 为首帧图片路径；留空则用当前默认（如 `train/ctx_5_20_per_frame_vae/image.png`）。
  - `dataset_frame`：`video_name` + `start_frame`，从数据集中取该帧作为首帧（运行脚本时会先导出为图再跑）。
- **recommended_pairs**（可选）：列出 `[prompt_id, first_chunk_id]`，只跑这些组合时可使用。

## 3. 推荐使用方式

1. **准备首帧图**（若用 fixed_image）：  
   将若干代表首帧（室内/室外/人脸等）放到固定路径，并在 `visual_eval_config.yaml` 的 `first_chunk_presets` 里填好 `path`。
2. **（可选）填 dataset_frame**：  
   在 `first_chunk_presets` 里填 `video_name`、`start_frame`，跑脚本时加 `--dataset_base /path/to/Context-as-Memory-Dataset`，会先导出该帧再跑。
3. **跑一组可视化**：  
   使用 `run_visual_eval.py`（见下），指定 `--ckpt`、`--evals_root`（或 `--output_root`），按 config 生成 2chunk/4chunk 到 `evals_visual/<prompt_id>_first_<first_chunk_id>/`。
4. **肉眼查看**：  
   打开对应目录，看 2chunk/4chunk 的 MP4：同一 prompt + 同一首帧下，对比不同 ckpt 的目录即可。

## 4. 运行示例

```bash
# 使用默认 config，跑所有 prompt × 首帧组合（首帧为 fixed 且 path 非空，或 dataset 已配置）
python -m eval_metrics.run_visual_eval \
  --ckpt /path/to/epoch-0.safetensors \
  --output_root /path/to/ckpt_dir/evals_visual \
  --config eval_metrics/visual_eval_config.yaml

# 只跑指定 prompt 与首帧
python -m eval_metrics.run_visual_eval \
  --ckpt /path/to/epoch-0.safetensors \
  --output_root /path/to/ckpt_dir/evals_visual \
  --prompts identity_single_person scene_indoor_room \
  --first_chunks fixed_default fixed_face

# 首帧来自数据集时提供 dataset_base
python -m eval_metrics.run_visual_eval \
  --ckpt /path/to/epoch-0.safetensors \
  --dataset_base /path/to/Context-as-Memory-Dataset \
  --output_root /path/to/ckpt_dir/evals_visual
```

## 5. 与现有 evals 的关系

- **回环 (1_loop_4chunk)**：首帧来自数据集随机采样 + 每段用该视频的 prompt，适合“随机多组、看分布”。
- **泛化 (2_generalization)**：单张固定首帧 + 单 prompt，已有 `run_generalization_fixed_first_frame.py`。
- **本设计**：在泛化流程上，把 **prompt 与首帧都做成可配置多组**，并**按 (prompt, 首帧) 分目录**，方便你系统性地“同一条件、多模型对比”或“同一模型、多条件查看”，以肉眼为主、metric 为辅。

## 6. 小结

- **Prompt**：在 `visual_eval_config.yaml` 的 `prompts` 里按身份/长时/回环/状态等设计多组文案。  
- **首 chunk**：在 `first_chunk_presets` 里配置固定图或 `dataset_frame`。  
- **查看**：用 `run_visual_eval.py` 按 config 生成到 `evals_visual/<prompt_id>_first_<id>/`，直接打开文件夹看 MP4 即可。
