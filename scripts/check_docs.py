"""Check local links, fragments and assets in the built documentation (stdlib only)."""
from __future__ import annotations

import argparse
import tomllib
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    def __init__(self, path: Path) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.links: list[tuple[str, int]] = []
        self.feed(path.read_text(encoding="utf-8"))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(values["id"])
        for attr in ("href", "src"):
            if values.get(attr):
                self.links.append((values[attr], self.getpos()[0]))


def check_site(root: Path, base_path: str = "/hailer/") -> list[str]:
    root = root.resolve()
    pages = {path: Page(path) for path in root.rglob("*.html")}
    if root / "index.html" not in pages:
        return [f"No built homepage in {root}; run zensical build first."]
    errors: list[str] = []
    for path, page in pages.items():
        for link, line in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            # Zensical's 404 page uses the canonical project prefix because it
            # can be served from an arbitrary missing URL on GitHub Pages.
            if url.path.startswith("/"):
                if not url.path.startswith(base_path):
                    errors.append(f"{path.relative_to(root)}:{line}: link escapes site prefix {link}")
                    continue
                target = (root / unquote(url.path[len(base_path):])).resolve()
            else:
                target = (path.parent / unquote(url.path)).resolve() if url.path else path
            if target.is_dir():
                target /= "index.html"
            if not target.is_relative_to(root) or not target.is_file():
                errors.append(f"{path.relative_to(root)}:{line}: missing target {link}")
            elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
                errors.append(f"{path.relative_to(root)}:{line}: missing fragment {link}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path, nargs="?", default=Path("site"))
    parser.add_argument("--config", type=Path, default=Path("zensical.toml"))
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    base_path = urlsplit(config["project"]["site_url"]).path.rstrip("/") + "/"
    errors = check_site(args.site, base_path)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"Checked {sum(1 for _ in args.site.rglob('*.html'))} HTML pages: local links, fragments and assets pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
