"""Run the real FineWeb2 precision filter before and after the TLD fix.

Example:
    python reproduce_tld_bug.py \
        --text "Ovo je jednostavan tekst na hrvatskom jeziku. Govori o svakodnevnom životu ljudi u Zagrebu, njihovom radu, učenju i druženju tijekom godine." \
        --url "https://example.hr/article"

The example text is classified as ``hrv_Latn`` by the pipeline's GlotLID
backend with a 0.9722 confidence score (the Croatian threshold is 0.467), while
containing zero entries from the official Croatian precision wordlist.

The production module starts SLURM jobs at import time, so this script uses the
same AST-based class loader as ``tests/test_url_filter.py``. The loaded
``Decontaminate`` classes and all filter method calls are otherwise unchanged.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
PIPELINE_PATH = SCRIPT_DIR / "run_precision_filtering.py"
BASE_REVISION = "d0defb24f193bb9a5a11b8b14524a03c4858e1b6"


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def load_filter_class(source: str, source_name: str):
    """Load only Decontaminate, without executing the module's SLURM jobs."""
    tree = ast.parse(source)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Decontaminate"
    )
    namespace = {}
    class_source = "".join(source.splitlines(keepends=True)[: class_node.end_lineno])
    exec(compile(class_source, source_name, "exec"), namespace)
    return namespace["Decontaminate"]


def source_at_revision(revision: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(REPOSITORY_ROOT),
            "show",
            f"{revision}:misc/precision_filtering/run_precision_filtering.py",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def make_filter(filter_class, whitelist: list[str]):
    return filter_class(
        "hrv",
        "hrv_Latn",
        whitelist_words=whitelist,
    )


def evaluate_pipeline(filter_instance, text: str, url: str) -> dict:
    """Call the real URL, wordlist, and combined pipeline methods."""
    url_document = SimpleNamespace(text=text, metadata={"url": url})
    wordlist_document = SimpleNamespace(text=text, metadata={"url": url})
    pipeline_document = SimpleNamespace(text=text, metadata={"url": url})

    url_passed = bool(filter_instance.url_filter(url_document))
    wordlist_passed = bool(filter_instance.wordlist_filter(wordlist_document))
    pipeline_passed = bool(filter_instance.filter(pipeline_document))

    from datatrove.utils.text import simplify_text, split_into_words

    normalized = simplify_text(text, filter_instance.norm_config)
    unique_tokens = set(split_into_words(normalized, filter_instance.language))
    matching_words = sorted(unique_tokens & filter_instance.wordlist())

    return {
        "url_passed": url_passed,
        "url_match": url_document.metadata.get("url_match"),
        "wordlist_passed": wordlist_passed,
        "wordlist_ratio": wordlist_document.metadata["wordlist_ratio"],
        "unique_token_count": len(unique_tokens),
        "matching_words": matching_words,
        "pipeline_passed": pipeline_passed,
    }


def run_comparison(text: str, url: str) -> tuple[dict, dict]:
    whitelist_data = json.loads((SCRIPT_DIR / "url_whitelist.json").read_text())
    whitelist = whitelist_data["hrv"]

    base_source = source_at_revision(BASE_REVISION)
    current_source = PIPELINE_PATH.read_text()
    base_class = load_filter_class(base_source, f"{BASE_REVISION}:{PIPELINE_PATH.name}")
    current_class = load_filter_class(current_source, str(PIPELINE_PATH))

    logging.info("Input text: %r", text)
    logging.info("Input URL: %s", url)
    logging.info("Official Croatian URL whitelist: %s", whitelist)
    logging.info("Calling Decontaminate from base revision %s", BASE_REVISION)
    logging.info("Calling Decontaminate from the current working tree")

    with working_directory(SCRIPT_DIR):
        base_result = evaluate_pipeline(make_filter(base_class, whitelist), text, url)
        current_result = evaluate_pipeline(
            make_filter(current_class, whitelist), text, url
        )
    return base_result, current_result


def decision(value: bool) -> str:
    return "ACCEPTED" if value else "REJECTED"


def print_report(base: dict, current: dict) -> None:
    matches = ", ".join(current["matching_words"]) or "(none)"
    bug_reproduced = not base["pipeline_passed"] and current["pipeline_passed"]

    print("\n=== FineWeb2 pipeline result ===")
    print(f"Unique tokens:             {current['unique_token_count']}")
    print(f"Wordlist match count:      {len(current['matching_words'])}")
    print(f"Wordlist matches:          {matches}")
    print(f"Wordlist ratio:            {current['wordlist_ratio']:.8f}")
    print()
    print(f"Base URL filter:           {decision(base['url_passed'])}")
    print(f"Current URL filter:        {decision(current['url_passed'])}")
    print(f"Current URL match:         {current['url_match'] or '(none)'}")
    print(f"Base pipeline decision:    {decision(base['pipeline_passed'])}")
    print(f"Current pipeline decision: {decision(current['pipeline_passed'])}")
    print(f"Bug reproduced:            {'YES' if bug_reproduced else 'NO'}")

    if current["wordlist_passed"]:
        print("Note: a wordlist hit accepts this document independently of its URL.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Call the real FineWeb2 Croatian precision pipeline before and after "
            "the regional-TLD fix."
        )
    )
    parser.add_argument("--text", required=True, help="Document text to filter.")
    parser.add_argument("--url", required=True, help="Document source URL.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if not args.url.strip():
        raise SystemExit("--url must not be empty")
    base_result, current_result = run_comparison(args.text, args.url)
    print_report(base_result, current_result)


if __name__ == "__main__":
    main()
