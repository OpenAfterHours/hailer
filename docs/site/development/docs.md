# Maintaining the docs

The site uses [Zensical](https://zensical.org/) with the OpenAfterHours palette and typography shared by
the RWA Calculator and Mooring sites. Zensical is pinned in the `docs` dependency group and `uv.lock`.
It is not a runtime dependency of Hailer.

## Preview and validate

From the repository root:

```bash
uv run --locked --group docs zensical serve
```

Open the local address printed by Zensical. Before submitting a change, build and check the generated links:

```bash
uv run --locked --group docs zensical build --strict
uv run --locked python scripts/check_docs.py
```

The checker verifies local HTML links, fragment identifiers and referenced assets. Check the landing page
at desktop and mobile widths, all three walkthrough buttons, keyboard focus, and the docs' search and theme toggle.
With JavaScript disabled, the first walkthrough step, chart and all documentation links remain visible.
The signal animation stops after two cycles and is disabled for reduced motion.

## Where things live

| Path | Purpose |
|---|---|
| `zensical.toml` | Site metadata, navigation, extensions and theme configuration |
| `docs/site/` | Published Markdown guides and assets |
| `docs/overrides/home.html` | Custom landing-page template |
| `docs/site/assets/stylesheets/tokens.css` | Vendored OpenAfterHours colours and font definitions |
| `docs/site/assets/stylesheets/theme.css` | Documentation shell styling |
| `docs/site/assets/stylesheets/landing.css` | Landing-page layout and responsive styles |
| `docs/site/assets/javascripts/landing.js` | Illustrated sample walkthrough controls |
| `docs/site/assets/examples/sales.csv` | Downloadable sample used by the walkthrough and quickstart |
| `site/` | Generated output, ignored by Git |

Public content is deliberately scoped to `docs/site/`. Engineering records such as
[LEARNINGS.md](https://github.com/OpenAfterHours/hailer/blob/main/docs/LEARNINGS.md),
[INTERFACES.md](https://github.com/OpenAfterHours/hailer/blob/main/docs/INTERFACES.md) and
[PLAN.md](https://github.com/OpenAfterHours/hailer/blob/main/PLAN.md) retain their existing paths.
The developer guides link to those records; the build does not publish the rest of `docs/`.

Edit the relevant guide when behaviour changes. Keep the README's concise quickstart aligned with the site's
[quickstart](../getting-started/quickstart.md). The sample walkthrough is an illustration, not a live model
connection; update its CSV, chart, summary and code together if the example changes.

## Publishing

`.github/workflows/docs.yml` builds and checks the site on pull requests. Pushes to `main` that affect the
site build and deploy it to <https://openafterhours.github.io/hailer/>. The workflow can also be run manually
from `main`. Pull requests never deploy.

For the first deployment, set the repository's **Settings → Pages → Build and deployment → Source** to
**GitHub Actions**. The `github-pages` environment must allow deployments from `main`.
Only the deploy job receives Pages and OIDC write permissions. The build installs the locked docs group alone,
so it does not need model credentials, Docker or Hailer's application dependencies.
