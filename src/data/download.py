"""Download ATUS files by resolving links from the official BLS year page.

File URLs are never hardcoded: we fetch https://www.bls.gov/tus/data/datafiles-<year>.htm,
collect its hrefs, and keep those matching the requested file kinds. Every downloaded file
is recorded in data/raw/manifest.csv (source URL, retrieval date, year, sha256, filename).
Raw files are written once and never modified.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import re
from pathlib import Path
from urllib.parse import urljoin

import requests

BLS_ROOT = "https://www.bls.gov"
YEAR_PAGE = BLS_ROOT + "/tus/data/datafiles-{year}.htm"

# BLS rejects requests without browser-like headers (HTTP 403).
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/18.0 Safari/605.1.15"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
}

# kind -> regex applied to the href's basename; {yy} / {year} filled per year
FILE_PATTERNS = {
    "activity": r"^atusact-{year}\.zip$",
    "respondent": r"^atusresp-{year}\.zip$",
    "roster": r"^atusrost-{year}\.zip$",
    "summary": r"^atussum-{year}\.zip$",
    "interview_dictionary": r"^atusintcodebk{yy}\.pdf$",
    "lexicon": r"^lexiconwex{year}\.(pdf|xls|xlsx)$",
    "coding_rules": r"^tu{year}coderules\.pdf$",
    "replicate_weights": r"^atuswgts-{year}\.zip$",
}
DEFAULT_KINDS = ["activity", "respondent", "roster", "interview_dictionary", "lexicon"]

MANIFEST_FIELDS = ["year", "kind", "filename", "source_url", "retrieved_utc", "sha256", "bytes"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_links(year: int, session: requests.Session) -> dict[str, list[str]]:
    page_url = YEAR_PAGE.format(year=year)
    resp = session.get(page_url, headers={**HEADERS, "Referer": BLS_ROOT + "/tus/data.htm"}, timeout=60)
    resp.raise_for_status()
    hrefs = re.findall(r'href="([^"]+)"', resp.text)
    found: dict[str, list[str]] = {}
    for kind, pat in FILE_PATTERNS.items():
        rx = re.compile(pat.format(year=year, yy=str(year)[-2:]), re.IGNORECASE)
        urls = sorted({urljoin(page_url, h) for h in hrefs if rx.match(h.rsplit("/", 1)[-1])})
        if urls:
            found[kind] = urls
    return found


def read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def append_manifest(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def download_year(year: int, raw_dir: Path, kinds: list[str] = DEFAULT_KINDS) -> list[dict]:
    """Download the requested kinds for one year. Existing files are skipped, not overwritten."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir = raw_dir / str(year)
    out_dir.mkdir(exist_ok=True)
    manifest = raw_dir / "manifest.csv"
    have = {r["filename"] for r in read_manifest(manifest)}

    session = requests.Session()
    links = resolve_links(year, session)
    missing = [k for k in kinds if k not in links]
    if missing:
        print(f"[{year}] WARNING: no link found on official page for: {missing}")

    rows = []
    for kind in kinds:
        for url in links.get(kind, []):
            fname = url.rsplit("/", 1)[-1]
            rel = f"{year}/{fname}"
            dest = out_dir / fname
            if dest.exists():
                print(f"[{year}] exists, skipping: {rel}")
                if rel not in have:
                    print(f"[{year}]   (not in manifest; run register to record its checksum)")
                continue
            r = session.get(url, headers={**HEADERS, "Referer": YEAR_PAGE.format(year=year)}, timeout=300)
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(r.content)
            tmp.rename(dest)
            row = {
                "year": year,
                "kind": kind,
                "filename": rel,
                "source_url": url,
                "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "sha256": sha256_file(dest),
                "bytes": dest.stat().st_size,
            }
            append_manifest(manifest, row)
            rows.append(row)
            print(f"[{year}] downloaded {rel} ({row['bytes']:,} bytes)")
    return rows


def register_existing(raw_dir: Path, year: int, source_url: str = "user-provided") -> None:
    """Record checksums for user-provided raw files that are not yet in the manifest."""
    manifest = raw_dir / "manifest.csv"
    have = {r["filename"] for r in read_manifest(manifest)}
    for p in sorted((raw_dir / str(year)).glob("*")):
        rel = f"{year}/{p.name}"
        if p.is_file() and rel not in have and not p.name.endswith(".part"):
            append_manifest(manifest, {
                "year": year, "kind": "user-provided", "filename": rel, "source_url": source_url,
                "retrieved_utc": dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc).isoformat(timespec="seconds"),
                "sha256": sha256_file(p), "bytes": p.stat().st_size,
            })
            print(f"registered {rel}")
