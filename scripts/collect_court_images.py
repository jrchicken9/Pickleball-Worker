from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Iterable

import requests
from PIL import Image

try:
    from ddgs import DDGS
except ImportError:  # fallback for older environments
    from duckduckgo_search import DDGS


DEFAULT_QUERIES = [
    "pickleball court",
    "indoor pickleball court",
    "outdoor pickleball court",
    "pickleball games court view",
    "pickleball championship court",
    "pickleball doubles match court",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect pickleball court images from web search."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/court_images",
        help="Directory where images and metadata are saved.",
    )
    parser.add_argument(
        "--max-per-query",
        type=int,
        default=120,
        help="Maximum candidate image URLs to fetch per query.",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=640,
        help="Minimum image width.",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=360,
        help="Minimum image height.",
    )
    parser.add_argument(
        "--sleep-ms",
        type=int,
        default=150,
        help="Delay between downloads to be polite to hosts.",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=None,
        help="Add a custom search query. Can be passed multiple times.",
    )
    return parser.parse_args()


def safe_ext_from_url(url: str) -> str:
    url_lower = url.lower()
    if ".png" in url_lower:
        return ".png"
    if ".webp" in url_lower:
        return ".webp"
    return ".jpg"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def iter_image_results(query: str, max_results: int) -> Iterable[dict]:
    with DDGS() as ddgs:
        try:
            try:
                # duckduckgo_search signature
                results = ddgs.images(
                    keywords=query,
                    max_results=max_results,
                )
            except TypeError:
                # ddgs signature
                results = ddgs.images(
                    query=query,
                    max_results=max_results,
                )
            for item in results:
                yield item
        except Exception as exc:  # noqa: BLE001
            print(f"  query failed ({query}): {exc}")
            return


def validate_image(path: Path, min_width: int, min_height: int) -> tuple[bool, str]:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            width, height = img.size
            if width < min_width or height < min_height:
                return False, f"too_small:{width}x{height}"
    except Exception as exc:
        return False, f"invalid_image:{exc}"
    return True, "ok"


def collect_images(
    output_dir: Path,
    queries: list[str],
    max_per_query: int,
    min_width: int,
    min_height: int,
    sleep_ms: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    seen_hashes: set[str] = set()
    existing = sorted(output_dir.glob("img_*.*"))
    next_id = len(existing) + 1

    if metadata_path.exists():
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                h = rec.get("sha256")
                if h:
                    seen_hashes.add(h)
            except json.JSONDecodeError:
                continue

    accepted = 0
    skipped = 0
    with metadata_path.open("a", encoding="utf-8") as mf:
        for query in queries:
            print(f"\n[query] {query}")
            count_for_query = 0
            for result in iter_image_results(query, max_per_query):
                image_url = result.get("image")
                source_url = result.get("url")
                if not image_url:
                    skipped += 1
                    continue

                try:
                    response = requests.get(image_url, timeout=20)
                    response.raise_for_status()
                    data = response.content
                except Exception:
                    skipped += 1
                    continue

                digest = sha256_bytes(data)
                if digest in seen_hashes:
                    skipped += 1
                    continue

                ext = safe_ext_from_url(image_url)
                file_path = output_dir / f"img_{next_id:06d}{ext}"
                file_path.write_bytes(data)
                ok, reason = validate_image(file_path, min_width, min_height)
                if not ok:
                    file_path.unlink(missing_ok=True)
                    skipped += 1
                    continue

                rec = {
                    "id": next_id,
                    "query": query,
                    "image_url": image_url,
                    "source_url": source_url,
                    "file": file_path.name,
                    "sha256": digest,
                    "status": "accepted",
                    "note": reason,
                }
                mf.write(json.dumps(rec, ensure_ascii=True) + "\n")
                mf.flush()

                seen_hashes.add(digest)
                accepted += 1
                count_for_query += 1
                next_id += 1

                if sleep_ms > 0:
                    time.sleep(sleep_ms / 1000.0)

            print(f"  accepted for query: {count_for_query}")

    print("\n=== Collection complete ===")
    print(f"accepted: {accepted}")
    print(f"skipped: {skipped}")
    print(f"output: {output_dir}")
    print(f"metadata: {metadata_path}")


def main() -> None:
    args = parse_args()
    queries = args.query if args.query else DEFAULT_QUERIES
    collect_images(
        output_dir=Path(args.output_dir),
        queries=queries,
        max_per_query=args.max_per_query,
        min_width=args.min_width,
        min_height=args.min_height,
        sleep_ms=args.sleep_ms,
    )


if __name__ == "__main__":
    main()
