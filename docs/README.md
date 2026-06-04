# Project page (local preview)

The **official** GitHub Pages site is deployed from the **`gh-pages`** branch (root `index.html` + `style.css`), not from this folder.

This `docs/` directory mirrors that pink-themed project page so you can preview locally:

```bash
cd docs
python -m http.server 18876 --bind 0.0.0.0
```

Then open `http://localhost:18876/` (with port forwarding if remote).

**Live site:** https://echo-team-joy-future-academy-jd.github.io/Echo-Memory/

To edit the public page, check out `gh-pages`, change `index.html` / `style.css` / `assets/`, and push to `origin gh-pages`.
