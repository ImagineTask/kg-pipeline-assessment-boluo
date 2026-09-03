"""Stage 1.2 step 1 - repeated header/footer detection.

Detection is by *position and repetition*, never by string match. The running
schedule header ("Joint Schedule 1 (Definitions)") is textually identical to the
legitimate heading that appears once at the true start of that schedule; a regex
on the string would delete real contract text. What separates them is that the
header repeats on nearly every page inside the top band, and the real heading
sits in the body and does not repeat.

One deviation from a naive reading of the spec. "Appears on >30% of *all* pages"
does not work here, because the running header *changes per schedule*: the
Joint Schedule 1 header covers 28 of 475 pages (5.9%) and would never clear a
global threshold. Repetition is therefore measured as density over the page span
the pattern actually occupies, which catches both per-schedule headers and the
document-wide ones.
"""
from __future__ import annotations

import re
from collections import defaultdict

from src.common import SETTINGS, load_jsonl, path, report

CFG = SETTINGS["boilerplate"]

SCHEDULE_HEADING = re.compile(
    r"^\s*(Core Terms"
    r"|Framework Award Form"
    r"|(?:Framework|Joint|Call[- ]?Off)\s+Schedule\s+\d+"
    r"|Annex\s+[A-Z0-9]+"
    r"|Appendix\s+\d+"
    r"|Part\s+[A-Z])\b",
    re.IGNORECASE,
)


def shape(text: str) -> str:
    """Collapse a block to a comparable shape: digits -> '#', whitespace normalised.

    Page numbers collapse onto one key, and so do footers whose only variation is
    a version number.
    """
    t = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\d", "#", t)


def block_key(block: dict) -> tuple[int, str] | None:
    if block.get("y_top") is None:
        return None
    y = int(round(block["y_top"] / CFG["y_round"]))
    return (y, shape(block["text"]))


def in_band(block: dict, page_height: float) -> str | None:
    """Top/bottom band membership, tested on y_top for both edges.

    Using the block *bottom* for the lower band would pull in the last body
    paragraph of a page, which is exactly the text we must not lose.
    """
    y_top = block.get("y_top")
    if y_top is None:
        return None
    if y_top <= page_height * CFG["top_band"]:
        return "top"
    if y_top >= page_height * (1 - CFG["bottom_band"]):
        return "bottom"
    return None


def _clusters(pages: list[int], max_gap: int = 3) -> list[list[int]]:
    """Split a pattern's page list into contiguous runs.

    Necessary because the same shape recurs in disjoint places: a bare page
    number "3" appears once per schedule, so its *global* span is the whole
    document and its global density is near zero. Density only means anything
    within a run.
    """
    out: list[list[int]] = []
    for p in pages:
        if out and p - out[-1][-1] <= max_gap:
            out[-1].append(p)
        else:
            out.append([p])
    return out


def _line_shapes(text: str) -> list[str]:
    return [shape(l) for l in text.splitlines() if l.strip()]


def detect(pages: list[dict]) -> tuple[dict, list[dict], dict[int, list[str]]]:
    """Return (blocks to strip as {(page, block_index)}, report rows).

    Two passes. The first finds repeated (position, shape) patterns by span
    density. The second generalises them to the *line* level, because Document AI
    inconsistently merges the two header lines into one block: on 23 pages of
    Joint Schedule 1 the header is one block, on 5 it is two. A block-shape key
    alone leaves those 5 pages with their header intact.
    """
    occurrences: dict[tuple[int, str], list[tuple[int, int, str]]] = defaultdict(list)
    bands: dict[tuple[int, str], set[str]] = defaultdict(set)

    for page in pages:
        h = page["page_height"]
        for i, block in enumerate(page["blocks"]):
            key = block_key(block)
            band = in_band(block, h)
            if key is None or band is None or not block["text"].strip():
                continue
            occurrences[key].append((page["page"], i, block["text"]))
            bands[key].add(band)

    rows: list[dict] = []
    boiler_lines: set[str] = set()

    for key, occ in occurrences.items():
        occ_sorted = sorted(occ)
        page_nums = sorted({p for p, _, _ in occ})
        y, sh = key
        for cluster in _clusters(page_nums):
            if len(cluster) < CFG["min_pages"]:
                continue
            span = cluster[-1] - cluster[0] + 1
            density = len(cluster) / span
            if density < CFG["min_density"]:
                continue
            sample = next(t for p, _, t in occ_sorted if p in set(cluster))
            boiler_lines.update(_line_shapes(sample))
            rows.append(
                {
                    "y_bucket": y,
                    "band": sorted(bands[key]),
                    "shape": sh[:140],
                    "sample_text": re.sub(r"\s+", " ", sample).strip()[:160],
                    "pages": len(cluster),
                    "page_span": [cluster[0], cluster[-1]],
                    "density": round(density, 3),
                }
            )

    # pass 2 - strip any banded block built purely from known boilerplate lines
    strip: set[tuple[int, int]] = set()
    kept_headings: set[str] = set()
    kept_blocks: set[tuple[int, int]] = set()
    kept: list[dict] = []
    page_headers: dict[int, list[str]] = {}
    for page in pages:
        h = page["page_height"]
        for i, block in enumerate(page["blocks"]):
            if in_band(block, h) is None:
                continue
            lines = _line_shapes(block["text"])
            if not lines or not all(l in boiler_lines for l in lines):
                continue
            if in_band(block, h) == "top":
                page_headers.setdefault(page["page"], []).append(block["text"].strip())
            head = SCHEDULE_HEADING.match(block["text"].strip())
            if head:
                label = shape(block["text"].splitlines()[0])
                if label not in kept_headings:
                    # keep the first occurrence at a genuine schedule boundary
                    kept_headings.add(label)
                    kept_blocks.add((page["page"], i))
                    kept.append({"page": page["page"], "text": label[:120]})
                    continue
            strip.add((page["page"], i))

    plan = {"drop": strip, "keep": kept_blocks, "lines": boiler_lines}
    rows.sort(key=lambda r: (-r["pages"], r["page_span"][0]))
    for r in rows:
        r["kept_first_at_schedule_boundary"] = any(
            k["text"].startswith(r["shape"].split(" Crown")[0][:40]) for k in kept
        )
    return plan, rows, page_headers


def main() -> None:
    pages = load_jsonl(path("raw_pages"))
    plan, rows, page_headers = detect(pages)
    strip = plan["drop"]
    total_blocks = sum(len(p["blocks"]) for p in pages)
    stripped_chars = sum(len(pages[p - 1]["blocks"][i]["text"]) for p, i in strip)
    total_chars = sum(len(b["text"]) for p in pages for b in p["blocks"])

    # every page should shed at least one header and one footer; a page that
    # sheds nothing usually means a layout the detector has not seen
    shed = defaultdict(int)
    for p, _ in strip:
        shed[p] += 1
    bare = [p["page"] for p in pages if shed[p["page"]] == 0]

    report(
        "boilerplate_report.json",
        {
            "pages": len(pages),
            "min_pages": CFG["min_pages"],
            "min_density": CFG["min_density"],
            "patterns_detected": len(rows),
            "blocks_stripped": len(strip),
            "blocks_total": total_blocks,
            "chars_stripped": stripped_chars,
            "chars_total": total_chars,
            "pct_chars_stripped": round(100 * stripped_chars / max(total_chars, 1), 2),
            "pages_with_nothing_stripped": bare,
            "pages_without_running_header": [p["page"] for p in pages if p["page"] not in page_headers],
            "patterns": rows,
        },
    )
    print(
        f"{len(rows)} boilerplate patterns, {len(strip)}/{total_blocks} blocks stripped "
        f"({100 * stripped_chars / max(total_chars, 1):.2f}% of characters); "
        f"{len(bare)} pages shed nothing"
    )


if __name__ == "__main__":
    main()
