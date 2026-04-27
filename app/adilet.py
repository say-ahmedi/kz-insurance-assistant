import logging
import re
from pathlib import Path

import httpx
from docx import Document

from .config import settings

log = logging.getLogger("adilet")

DEFAULT_TIMEOUT = 30
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; insurance-assistant/1.0)",
    "Accept-Language": "ru,en;q=0.8",
}


def _strip_html(html):
    html = re.sub(r"(?is)<script.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?</style>", " ", html)
    html = re.sub(r"(?is)<br\s*/?>", "\n", html)
    html = re.sub(r"(?is)</p>", "\n\n", html)
    html = re.sub(r"(?is)<[^>]+>", "", html)
    html = re.sub(r"[ \t]+", " ", html)
    html = re.sub(r"\n{3,}", "\n\n", html)
    return html.strip()


def fetch_law(url, save_as=None, laws_dir=None):
    """Download a law page from adilet.zan.kz and store it as a .docx in laws/.

    Returns the saved path. The result still has to be re-indexed via
    /admin/reindex (or python -m app.ingest) before it's queryable.
    """
    if "adilet.zan.kz" not in url:
        raise ValueError("Only adilet.zan.kz URLs are accepted")

    laws_dir = Path(laws_dir or settings.laws_dir)
    laws_dir.mkdir(parents=True, exist_ok=True)

    log.info("Fetching %s", url)
    r = httpx.get(url, headers=HEADERS, timeout=DEFAULT_TIMEOUT, follow_redirects=True)
    r.raise_for_status()
    text = _strip_html(r.text)
    if len(text) < 500:
        raise RuntimeError("Page looks empty — login or captcha may be required")

    if not save_as:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", url.split("/")[-1]) or "adilet_doc"
        save_as = f"adilet_{slug}.docx"
    if not save_as.lower().endswith(".docx"):
        save_as += ".docx"

    target = laws_dir / save_as
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    doc.save(target)

    log.info("Saved %s (%d chars)", target, len(text))
    return target
