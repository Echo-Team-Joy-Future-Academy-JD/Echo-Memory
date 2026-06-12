# Developer Guide / 开发者指南

Bilingual interactive version: [Project Page → Developer Guide](https://echo-team-joy-future-academy-jd.github.io/Echo-Memory/developer.html)

Hands-on development, training/eval workflows, and **Cursor Agent skills** for Echo-Memory.

实战开发、训练评测与 **Cursor Agent 技能**。

---

## Cursor skills / 项目 Skills

Project skills live in **`.cursor/skills/`** — reference them in Agent chat (e.g. *use echo-memory-eval to …*).

| Skill | English | 中文 |
| --- | --- | --- |
| `echo-memory-agent` | Scope prompts, rules, skill index | Prompt 范围、Rules、技能索引 |
| `echo-memory-train` | Memory baselines & context training | Baseline 与 Context 训练 |
| `echo-memory-eval` | Replay / revisit & HF checkpoint checks | 回放 / revisit、HF checkpoint check |
| `echo-memory-release` | gh-pages, i18n, checkpoints doc | gh-pages、i18n、权重文档 |

Index: [.cursor/skills/README.md](../.cursor/skills/README.md)

---

## 1. Guide map / 文档地图

| | English | 中文 |
| --- | --- | --- |
| **README** | Paper overview, quick start, checkpoints, community | 论文概览、快速上手、权重、社区 |
| **This guide** | Workflows, Cursor skills, Agent tips | 工作流、Skills、Agent 技巧 |
| **`doc/`** | Dataset & checkpoint reference | 数据集与权重参考 |

---

## 2. Environment & paths / 环境与路径

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=data/Context-as-Memory-Dataset
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export OUTPUT_BASE_ROOT=$PWD/outputs
```

| Pool | English | 中文 |
| --- | --- | --- |
| Static in-domain | Default root above — [dataset_preprocessing.md](dataset_preprocessing.md) | 默认路径 — 同上 |
| Dynamic training | e.g. `data/dynamic-memory-dataset` — [dynamic_dataset_preprocessing.md](dynamic_dataset_preprocessing.md) | 如 `data/dynamic-memory-dataset` — 同上 |
| Checkpoints | [Echo-Team/Echo-Memory](https://huggingface.co/Echo-Team/Echo-Memory) — [checkpoints.md](checkpoints.md) | 同上 |

---

## 3. Code map / 代码地图

| Path | English | 中文 |
| --- | --- | --- |
| `.cursor/skills/` | Cursor Agent skills | Agent 技能 |
| `train/memory_baselines_basic/` | Spatial / SSM / compression | Spatial / SSM / 压缩 |
| `train/context_learning/` | Context K=1/5/20 | Context 配方 |
| `eval/v2/` | Replay, revisit | 回放、revisit |
| `env/memory_baseline_runtime.py` | CKPT → memory profile | 权重 → 记忆配置 |
| `docs/` | GitHub Pages | 项目页 |

---

## 4. Common workflows / 常用工作流

**Train / 训练**

```bash
bash train/memory_baselines_basic/run_spatial_memory_baseline.sh
bash train/context_learning/run_pre_qkv_ctx20.sh
```

**Checkpoint eval / checkpoint 检查**

```bash
huggingface-cli download Echo-Team/Echo-Memory spatial_mem/epoch-0.safetensors --local-dir ./ckpts
export CKPT=./ckpts/spatial_mem/epoch-0.safetensors
bash eval/v2/run_static_consistency_loop_and_revisit.sh
```

Keep the row folder name in `CKPT`.

---

## 5. Agent prompts / 示例 Prompt

```text
Using echo-memory-eval: download context_k1 from Echo-Team/Echo-Memory
and run eval/v2/run_basic_replay_gt.sh with the static in-domain pool.

Using echo-memory-train: document OUTPUT_BASE_ROOT override in
run_ablation_block_wise_ssm_two_chunk.sh.
```

**Public repo hygiene / 公开仓库规范:** no upload bash, internal benchmark names, or machine paths in GitHub.

---

## 6. Site & release / 站点与发布

```bash
bash scripts/publish_gh_pages.sh
```

Community QR: [project page → Updates](https://echo-team-joy-future-academy-jd.github.io/Echo-Memory/#updates) or README **Community**.

---

## 7. Checklist / 检查清单

- [ ] Quick eval with one HF checkpoint
- [ ] `doc/checkpoints.md` matches HF folders
- [ ] Public docs use Echo pool names
- [ ] Publish gh-pages after site edits; verify EN/中文 toggle
