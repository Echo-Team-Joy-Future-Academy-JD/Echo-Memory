# Developer Guide / 开发者指南

Bilingual interactive version: [Project Page → Developer Guide](https://echo-team-joy-future-academy-jd.github.io/Echo-Memory/developer.html)

Hands-on development, training/eval workflows, and **Cursor vibe coding** for Echo-Memory.  
This is **not** a copy of the public README (paper story, community QR → see README & project page).

实战开发、训练评测与 **Cursor 协作编程**。  
**不是** README 重复版（论文叙事、社区二维码 → 见 README 与项目页）。

---

## 1. What this guide is / 本指南定位

| | English | 中文 |
| --- | --- | --- |
| **README** | Paper story, quick start, checkpoints, community QR | 论文叙事、快速上手、权重、社区二维码 |
| **This guide** | Code map, train/eval, Cursor Agent tips | 代码地图、训练/评测、Cursor 协作 |
| **`doc/`** | Dataset & checkpoint deep dives | 数据集与权重细节 |

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
| Static in-domain | Default root above — [dataset_preprocessing.md](dataset_preprocessing.md) | 默认路径见上 — 同上 |
| Dynamic training | e.g. `data/dynamic-memory-dataset` — [dynamic_dataset_preprocessing.md](dynamic_dataset_preprocessing.md) | 如 `data/dynamic-memory-dataset` — 同上 |
| Checkpoints | [Echo-Team/Echo-Memory](https://huggingface.co/Echo-Team/Echo-Memory) — [checkpoints.md](checkpoints.md) | 同上 |

---

## 3. Code map / 代码地图

| Path | English | 中文 |
| --- | --- | --- |
| `train/memory_baselines_basic/` | Spatial / SSM / compression ablations | Spatial / SSM / 压缩消融 |
| `train/context_learning/` | Context K=1/5/20 | Context K=1/5/20 |
| `eval/v2/` | Replay, loop closure, open-domain revisit | 回放、闭环、开放域 revisit |
| `env/memory_baseline_runtime.py` | Checkpoint → memory profile | 权重 → 记忆配置 |
| `diffsynth/` | Wan backbone & training stack | Wan 骨干与训练栈 |
| `docs/` | GitHub Pages site | GitHub Pages 站点 |

---

## 4. Common workflows / 常用工作流

**Train one row / 训练一行**

```bash
bash train/memory_baselines_basic/run_spatial_memory_baseline.sh
bash train/context_learning/run_pre_qkv_ctx20.sh
```

**Smoke eval / smoke 评测**

```bash
huggingface-cli download Echo-Team/Echo-Memory spatial_mem/epoch-0.safetensors --local-dir ./ckpts
export CKPT=./ckpts/spatial_mem/epoch-0.safetensors
bash eval/v2/run_static_consistency_loop_and_revisit.sh
```

Keep the row folder name in `CKPT` / `CKPT` 路径需保留行目录名。

---

## 5. Cursor vibe coding / Cursor 协作编程

Use **Cursor Agent** (Composer) for multi-file work.

| Tip | English | 中文 |
| --- | --- | --- |
| Scope | Name memory family, script, eval branch (replay / in-domain / open-domain) | 写清 memory 家族、脚本、评测分支 |
| Entry points | e.g. `run_spatial_memory_baseline.sh`, `eval/v2/run_basic_replay_gt.sh` | 指向具体入口脚本 |
| Rules | `.cursor/rules/echo-memory.mdc` — pool naming, public doc constraints | 池命名、公开文档约束 |
| Skills | `create-rule`, `create-skill`, `babysit` for release loops | 规则/技能/CI 循环 |
| Ask mode | Read `diffsynth/` or trace checkpoint mapping without edits | 只读追踪、不改代码 |

**Example prompt / 示例 Prompt**

```text
Add a smoke test that downloads context_k1 from Echo-Team/Echo-Memory
and runs eval/v2/run_basic_replay_gt.sh with the static in-domain pool.

Trace env/memory_baseline_runtime.py spatial_mem → inject flags;
summarize in doc/checkpoints.md.
```

**Public repo hygiene / 公开仓库规范:** no upload bash, internal benchmark names, or machine paths in GitHub. WeChat QR → project page & README only, **not** this guide.

---

## 6. Site & release (maintainers) / 站点与发布

```bash
bash scripts/publish_gh_pages.sh
```

- HF weights: Hugging Face UI or `hf upload` (maintainers only)
- Bilingual project page: `docs/i18n.js` + `docs/i18n-runtime.js`
- 权重：网页或 `hf upload`（仅维护者）；项目页双语见 `docs/i18n*.js`

---

## 7. Checklist / 检查清单

- [ ] Smoke eval with one HF checkpoint / 至少一个 HF 权重 smoke eval
- [ ] `doc/checkpoints.md` matches HF folders / 与 HF 目录一致
- [ ] Public docs use Echo pool names / 公开文档用 Echo 池命名
- [ ] Publish gh-pages after site edits / 改站点后发布并检查 EN/中文

Community QR / 社区二维码: [project page → Updates](https://echo-team-joy-future-academy-jd.github.io/Echo-Memory/#updates) or README **Community** section.
