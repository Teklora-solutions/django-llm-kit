"""MkDocs hook: publish the docs for AI tools as well as people.

For every page in the nav, the Markdown source is copied next to the HTML
(``guides/agents/`` is also served as ``guides/agents.md``), and two indexes are
written at the site root, following https://llmstxt.org:

* ``llms.txt``: the site summary and a link list of every page, with its description.
* ``llms-full.txt``: every page's Markdown in nav order, in one file.

Each HTML page also links its Markdown twin (``<link rel="alternate" type="text/markdown">``,
added by the theme override) so agents that fetch the HTML can find the source.
"""

from __future__ import annotations

import re
from pathlib import Path

_FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n+", re.S)
_pages: list = []
_order: dict[str, int] = {}


def on_pre_build(config, **kwargs):
    _pages.clear()
    _order.clear()


def on_nav(nav, config, files, **kwargs):
    _order.update({page.file.src_uri: i for i, page in enumerate(nav.pages)})
    return nav


def on_page_markdown(markdown, page, config, files, **kwargs):
    _pages.append((page, _FRONT_MATTER.sub("", markdown)))
    return markdown


def _section_of(page) -> str:
    parent = page.parent
    return parent.title if parent is not None else "Docs"


def on_post_build(config, **kwargs):
    site = Path(config["site_dir"])
    base = config["site_url"].rstrip("/") + "/" if config.get("site_url") else ""
    pages = sorted(_pages, key=lambda item: _order.get(item[0].file.src_uri, len(_order)))

    sections: dict[str, list[str]] = {}
    full = [f"# {config['site_name']}\n\n> {config['site_description']}\n"]
    for page, markdown in pages:
        src = page.file.src_uri
        target = site / src
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(markdown, encoding="utf-8")

        desc = (page.meta or {}).get("description", "")
        line = f"- [{page.title}]({base}{src})" + (f": {desc}" if desc else "")
        sections.setdefault(_section_of(page), []).append(line)
        full.append(f"\n\n---\n\n<!-- source: {base}{src} -->\n\n{markdown.strip()}\n")

    out = [f"# {config['site_name']}", "", f"> {config['site_description']}", ""]
    out.append(
        "Every page below is plain Markdown. The whole documentation in one file: "
        f"[llms-full.txt]({base}llms-full.txt). Source code: {config.get('repo_url', '')}"
    )
    for title, lines in sections.items():
        out += ["", f"## {title}", "", *lines]
    (site / "llms.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    (site / "llms-full.txt").write_text("".join(full), encoding="utf-8")
