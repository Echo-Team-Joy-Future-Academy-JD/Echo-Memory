---
name: echo-memory-release
description: >-
  Publish Echo-Memory GitHub Pages, update bilingual docs/i18n, and align
  doc/checkpoints.md with Hugging Face Echo-Team/Echo-Memory. Use for gh-pages
  releases, project page edits, or maintainer checkpoint documentation.
---

# Echo-Memory release & site

## GitHub Pages

Source: `docs/` → branch `gh-pages`

```bash
bash scripts/publish_gh_pages.sh
```

Touch `docs/index.html`, `docs/style.css`, `docs/site.js`, `docs/i18n.js`, `docs/i18n-runtime.js`, `docs/developer.html` as needed.

- Language toggle: `docs/i18n.js` + `docs/i18n-runtime.js`, key `echo-memory-lang`
- Developer guide: `docs/developer.html` (EN/中文 toggle via same i18n)

## Hugging Face checkpoints

- Repo: **Echo-Team/Echo-Memory**
- Layout: `{row_id}/epoch-0.safetensors`
- Public index: `doc/checkpoints.md` + README **Checkpoints** table

Update weights via Hugging Face UI or `hf upload` (maintainers only). **Do not** add upload bash scripts to the public GitHub repo.

## Public documentation rules

Use Echo pool names in GitHub-facing markdown:

| Use | Avoid in public docs |
| --- | --- |
| Static in-domain pool | Internal codenames, private benchmark names |
| Dynamic training pool | Machine-local absolute paths |

WeChat QR: `assets/wechat_group_qrcode.jpg` + project page Community section only.

## Release checklist

- [ ] Smoke eval one HF checkpoint
- [ ] `doc/checkpoints.md` matches HF folder names
- [ ] `publish_gh_pages.sh` if site changed; verify EN/中文 on live page
- [ ] News section on project page if user-visible release
