from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.robotparser
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

try:
    import spacy
except ImportError:
    spacy = None


DEFAULT_USER_AGENT = "NERResearchCrawler/1.0"
DEFAULT_TIMEOUT = 15
DEFAULT_DELAY = 1.0
DEFAULT_MAX_CHARS = 300_000

SKIP_EXTENSIONS = {
    ".7z", ".avi", ".bin", ".bmp", ".bz2", ".css", ".csv", ".doc", ".docx",
    ".eot", ".exe", ".gif", ".gz", ".ico", ".iso", ".jpeg", ".jpg", ".js",
    ".json", ".m4a", ".m4v", ".mov", ".mp3", ".mp4", ".mpeg", ".mpg", ".odt",
    ".ogg", ".otf", ".pdf", ".png", ".ppt", ".pptx", ".rar", ".rss", ".svg",
    ".tar", ".tgz", ".tif", ".tiff", ".ttf", ".wav", ".webm", ".webp", ".woff",
    ".woff2", ".xls", ".xlsx", ".xml", ".xz", ".zip",
}

DEFAULT_LABELS = {
    "PERSON", "ORG", "GPE", "LOC", "FAC", "NORP",
    "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
    "DATE", "TIME", "MONEY", "PERCENT", "QUANTITY",
    "ORDINAL", "CARDINAL",
}


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int


@dataclass
class PageResult:
    url: str
    title: str
    depth: int
    status_code: int
    text_chars: int
    entities: list[Entity]
    text: str | None = None


def normalize_url(url: str) -> str:
    """Drop fragments and normalize obvious URL noise."""
    p = urlparse(url)
    scheme = p.scheme.lower()
    netloc = p.netloc.lower()

    # Keep the path/query because they can identify distinct pages.
    path = p.path or "/"

    # Drop fragments.
    return urlunparse((scheme, netloc, path, "", p.query, ""))


def is_http_url(url: str) -> bool:
    return urlparse(url).scheme.lower() in {"http", "https"}


def same_hostname(seed_host: str, url: str) -> bool:
    return (urlparse(url).hostname or "").lower() == seed_host.lower()


def looks_like_html_link(url: str) -> bool:
    path = (urlparse(url).path or "").lower()
    return not any(path.endswith(ext) for ext in SKIP_EXTENSIONS)


def clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_visible_text_and_links(html: str, base_url: str) -> tuple[str, str, list[str]]:
    soup = BeautifulSoup(html, "html.parser")

    title = clean_whitespace(soup.title.get_text(" ", strip=True)) if soup.title else ""

    # Remove content that is generally not useful for page-level NER.
    for tag in soup(
        [
            "script", "style", "noscript", "template", "svg",
            "canvas", "iframe",
        ]
    ):
        tag.decompose()

    # Prefer semantic page content when available.
    root = soup.find("main") or soup.find("article") or soup.body or soup
    text = clean_whitespace(root.get_text(" ", strip=True))

    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:", "data:")):
            continue

        absolute = normalize_url(urljoin(base_url, href))
        if is_http_url(absolute):
            links.append(absolute)

    return title, text, links


class RobotsCache:
    def __init__(self, session: requests.Session, user_agent: str, timeout: int):
        self.session = session
        self.user_agent = user_agent
        self.timeout = timeout
        self._cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    def _origin(self, url: str) -> str:
        p = urlparse(url)
        return f"{p.scheme}://{p.netloc}"

    def _load(self, url: str):
        origin = self._origin(url)
        if origin in self._cache:
            return self._cache[origin]

        robots_url = f"{origin}/robots.txt"
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(robots_url)

        try:
            response = self.session.get(
                robots_url,
                headers={"User-Agent": self.user_agent},
                timeout=self.timeout,
            )

            # If robots.txt exists, parse it. If it is missing, allow crawl.
            if response.status_code == 200:
                parser.parse(response.text.splitlines())
                self._cache[origin] = parser
            elif response.status_code in {401, 403}:
                # Conservative handling: treat explicit access denial as disallow.
                deny = urllib.robotparser.RobotFileParser()
                deny.parse(["User-agent: *", "Disallow: /"])
                self._cache[origin] = deny
            else:
                self._cache[origin] = None

        except requests.RequestException:
            # robots.txt could not be obtained. We don't fabricate rules.
            self._cache[origin] = None

        return self._cache[origin]

    def allowed(self, url: str) -> bool:
        parser = self._load(url)
        if parser is None:
            return True
        return parser.can_fetch(self.user_agent, url)

    def crawl_delay(self, url: str) -> float | None:
        parser = self._load(url)
        if parser is None:
            return None

        delay = parser.crawl_delay(self.user_agent)
        if delay is None:
            delay = parser.crawl_delay("*")

        return float(delay) if delay is not None else None


class NERScraper:
    def __init__(
        self,
        start_url: str,
        model: str,
        max_pages: int,
        max_depth: int,
        delay: float,
        timeout: int,
        max_chars: int,
        labels: set[str] | None,
        include_text: bool,
        user_agent: str,
    ):
        self.start_url = normalize_url(start_url)
        self.seed_host = (urlparse(self.start_url).hostname or "").lower()

        self.max_pages = max_pages
        self.max_depth = max_depth
        self.delay = delay
        self.timeout = timeout
        self.max_chars = max_chars
        self.labels = labels
        self.include_text = include_text
        self.user_agent = user_agent

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
                "Accept-Language": "en-US,en;q=0.8",
            }
        )

        self.robots = RobotsCache(self.session, user_agent, timeout)
        self.nlp = self._load_model(model)

    @staticmethod
    def _load_model(model: str):
        if spacy is None:
            raise RuntimeError(
                "spaCy is not installed. Run: pip install spacy"
            )
        try:
            return spacy.load(model)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{model}' is not installed.\n"
                f"Install it with:\n"
                f"    python -m spacy download {model}"
            ) from exc

    def _fetch(self, url: str) -> requests.Response | None:
        if not self.robots.allowed(url):
            print(f"[robots] blocked: {url}", file=sys.stderr)
            return None

        robots_delay = self.robots.crawl_delay(url)
        sleep_for = max(self.delay, robots_delay or 0.0)
        if sleep_for > 0:
            time.sleep(sleep_for)

        try:
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"[fetch] {url}: {exc}", file=sys.stderr)
            return None

        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
            print(f"[skip] non-HTML: {url} ({content_type or 'unknown type'})", file=sys.stderr)
            return None

        return response

    def _extract_entities(self, text: str) -> list[Entity]:
        # Hard cap protects memory / NLP runtime against unexpectedly huge pages.
        bounded = text[: self.max_chars]
        doc = self.nlp(bounded)

        entities: list[Entity] = []
        for ent in doc.ents:
            label = ent.label_
            value = clean_whitespace(ent.text)

            if not value:
                continue
            if self.labels and label not in self.labels:
                continue

            entities.append(
                Entity(
                    text=value,
                    label=label,
                    start=ent.start_char,
                    end=ent.end_char,
                )
            )

        return entities

    def crawl(self) -> list[PageResult]:
        queue = deque([(self.start_url, 0)])
        queued = {self.start_url}
        visited: set[str] = set()
        pages: list[PageResult] = []

        while queue and len(pages) < self.max_pages:
            url, depth = queue.popleft()
            queued.discard(url)

            if url in visited:
                continue
            visited.add(url)

            if not same_hostname(self.seed_host, url):
                continue
            if not looks_like_html_link(url):
                continue

            print(
                f"[{len(pages) + 1}/{self.max_pages}] depth={depth} {url}",
                file=sys.stderr,
            )

            response = self._fetch(url)
            if response is None:
                continue

            final_url = normalize_url(response.url)
            if not same_hostname(self.seed_host, final_url):
                print(f"[skip] redirected off-host: {final_url}", file=sys.stderr)
                continue

            title, text, links = extract_visible_text_and_links(
                response.text,
                final_url,
            )

            if not text:
                continue

            entities = self._extract_entities(text)

            pages.append(
                PageResult(
                    url=final_url,
                    title=title,
                    depth=depth,
                    status_code=response.status_code,
                    text_chars=len(text),
                    entities=entities,
                    text=text[: self.max_chars] if self.include_text else None,
                )
            )

            if depth >= self.max_depth:
                continue

            for link in links:
                if len(visited) + len(queue) >= self.max_pages * 20:
                    # Prevent pathological pages from exploding the queue.
                    break

                if (
                    same_hostname(self.seed_host, link)
                    and looks_like_html_link(link)
                    and link not in visited
                    and link not in queued
                ):
                    queue.append((link, depth + 1))
                    queued.add(link)

        return pages


def aggregate_entities(pages: Iterable[PageResult]) -> list[dict]:
    counts: Counter[tuple[str, str]] = Counter()
    urls_by_entity: dict[tuple[str, str], set[str]] = defaultdict(set)

    for page in pages:
        for ent in page.entities:
            key = (ent.text, ent.label)
            counts[key] += 1
            urls_by_entity[key].add(page.url)

    rows = []
    for (text, label), count in counts.most_common():
        urls = sorted(urls_by_entity[(text, label)])
        rows.append(
            {
                "text": text,
                "label": label,
                "count": count,
                "page_count": len(urls),
                "urls": urls,
            }
        )
    return rows


def write_outputs(
    pages: list[PageResult],
    output_dir: Path,
    start_url: str,
    include_text: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate = aggregate_entities(pages)

    json_path = output_dir / "ner_results.json"
    csv_path = output_dir / "entities.csv"
    occurrences_path = output_dir / "entity_occurrences.csv"

    payload = {
        "start_url": start_url,
        "pages_crawled": len(pages),
        "unique_entities": len(aggregate),
        "entities": aggregate,
        "pages": [
            {
                **{
                    "url": p.url,
                    "title": p.title,
                    "depth": p.depth,
                    "status_code": p.status_code,
                    "text_chars": p.text_chars,
                    "entities": [asdict(e) for e in p.entities],
                },
                **({"text": p.text} if include_text else {}),
            }
            for p in pages
        ],
    }

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["text", "label", "count", "page_count", "urls"],
        )
        writer.writeheader()
        for row in aggregate:
            writer.writerow(
                {
                    **row,
                    "urls": " | ".join(row["urls"]),
                }
            )

    with occurrences_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["url", "title", "entity", "label", "start", "end"],
        )
        writer.writeheader()
        for page in pages:
            for ent in page.entities:
                writer.writerow(
                    {
                        "url": page.url,
                        "title": page.title,
                        "entity": ent.text,
                        "label": ent.label,
                        "start": ent.start,
                        "end": ent.end,
                    }
                )

    return json_path, csv_path, occurrences_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Same-host web crawler + spaCy named-entity extraction."
    )
    parser.add_argument("url", help="Starting http(s) URL.")
    parser.add_argument(
        "--model",
        default="en_core_web_sm",
        help="spaCy model name (default: en_core_web_sm).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=20,
        help="Maximum number of successfully processed pages (default: 20).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Maximum link depth from seed page (default: 1).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help="Minimum seconds between requests (default: 1.0).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds (default: 15).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="Maximum characters sent to NER per page.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help=(
            "Optional spaCy entity labels to keep, e.g. "
            "--labels PERSON ORG GPE. Default: keep all model-produced labels."
        ),
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include extracted page text in ner_results.json.",
    )
    parser.add_argument(
        "--output",
        default="ner_output",
        help="Output directory (default: ner_output).",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help=f"HTTP User-Agent (default: {DEFAULT_USER_AGENT}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not is_http_url(args.url):
        print("URL must begin with http:// or https://", file=sys.stderr)
        return 2

    if args.max_pages < 1:
        print("--max-pages must be >= 1", file=sys.stderr)
        return 2
    if args.depth < 0:
        print("--depth must be >= 0", file=sys.stderr)
        return 2
    if args.delay < 0:
        print("--delay must be >= 0", file=sys.stderr)
        return 2
    if args.max_chars < 1:
        print("--max-chars must be >= 1", file=sys.stderr)
        return 2

    labels = set(args.labels) if args.labels else None

    try:
        scraper = NERScraper(
            start_url=args.url,
            model=args.model,
            max_pages=args.max_pages,
            max_depth=args.depth,
            delay=args.delay,
            timeout=args.timeout,
            max_chars=args.max_chars,
            labels=labels,
            include_text=args.include_text,
            user_agent=args.user_agent,
        )
        pages = scraper.crawl()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    json_path, csv_path, occurrences_path = write_outputs(
        pages=pages,
        output_dir=Path(args.output),
        start_url=scraper.start_url,
        include_text=args.include_text,
    )

    print(f"\nCrawled pages: {len(pages)}")
    print(f"JSON: {json_path}")
    print(f"Aggregate CSV: {csv_path}")
    print(f"Occurrence CSV: {occurrences_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
