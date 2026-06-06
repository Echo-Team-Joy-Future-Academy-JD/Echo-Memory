# Developer Guide / 开发者手册

Maintainer notes for [Echo-Memory](https://github.com/Echo-Team-Joy-Future-Academy-JD/Echo-Memory).  
Interactive bilingual version: [Project Page → Developer Guide](https://echo-team-joy-future-academy-jd.github.io/Echo-Memory/developer.html)

维护说明（中英双语交互版见上方链接）。

---

## 1. Repository layout / 仓库结构

| Path | Purpose |
| --- | --- |
| `docs/` | GitHub Pages site (`index.html`, `developer.html`, `i18n.js`) |
| `doc/` | Dataset & checkpoint markdown |
| `train/` | Training recipes |
| `eval/v2/` | Replay & revisit evaluation |
| `assets/` | README figures, WeChat QR (`wechat_group_qrcode.jpg`) |
| `scripts/publish_gh_pages.sh` | Publish `docs/` → `gh-pages` branch |

---

## 2. Publish project page / 发布项目页

```bash
cd Echo-Memory
bash scripts/publish_gh_pages.sh
```

Live site: https://echo-team-joy-future-academy-jd.github.io/Echo-Memory/

After editing `docs/index.html`, `docs/style.css`, `docs/site.js`, or `docs/i18n*.js`, run the script from `main` and verify EN/中文 toggle.

---

## 3. Hugging Face checkpoints / 更新权重

**Repo:** [Echo-Team/Echo-Memory](https://huggingface.co/Echo-Team/Echo-Memory)

- Index: [doc/checkpoints.md](checkpoints.md)
- Layout: `{row_id}/epoch-0.safetensors`
- Update model card README via Hugging Face UI or `hf upload` (maintainers only; upload bash scripts are **not** in the public GitHub repo)

```bash
hf auth login   # Echo-Team org write access
hf upload Echo-Team/Echo-Memory ./ckpts/spatial_mem/epoch-0.safetensors spatial_mem/epoch-0.safetensors
```

---

## 4. Bilingual site / 中英双语

- Strings: `docs/i18n.js`
- Runtime: `docs/i18n-runtime.js`
- Mark elements with `data-i18n`, `data-i18n-html`, or `data-i18n-attr`
- Preference key: `localStorage['echo-memory-lang']` → `en` or `zh`

---

## 5. WeChat group / 微信群

Replace both copies when the QR code refreshes. Use the **full WeChat group screenshot** as-is (no manual crop):

- `docs/assets/wechat_group_qrcode.jpg` (project page)
- `assets/wechat_group_qrcode.jpg` (README)

Then republish gh-pages. Pages scale the portrait image proportionally (`height: auto`).

<div align="center">
<img src="../assets/wechat_group_qrcode.jpg" alt="Echo-Memory WeChat group" width="1166" height="1640" style="width:260px;height:auto;max-width:100%;">
<p><b>Echo-Memory 交流群</b></p>
</div>

---

## 6. Release checklist / 发布前检查

- [ ] Smoke-test eval with one HF checkpoint
- [ ] `doc/checkpoints.md` matches HF folder names
- [ ] `bash scripts/publish_gh_pages.sh` — verify live EN/ZH
- [ ] Update project page News if user-visible
