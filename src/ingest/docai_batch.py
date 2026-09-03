"""Stage 1.1 - Document AI batch OCR.

Batch (asynchronous) processing is mandatory here: the document is 475 pages and
online processing caps out far below that. The PDF is sharded before upload so we
never sit near a per-request page limit, and every shard carries a *global page
offset* so page numbers survive the merge. An off-by-N offset silently corrupts
every citation downstream, so the offset is derived from the shard manifest and
asserted after the merge.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as documentai
from google.cloud import storage
from pypdf import PdfReader, PdfWriter

from src.common import ROOT, SETTINGS, env, path, report, write_jsonl
from src.gcp_auth import credentials

SHARD_PAGES = 100  # comfortably inside the Enterprise Document OCR batch limit


# --------------------------------------------------------------------------- #
# sharding
# --------------------------------------------------------------------------- #
def shard_pdf(pdf_path: Path, out_dir: Path, shard_pages: int = SHARD_PAGES) -> list[dict]:
    """Split the PDF into fixed-size shards. Returns a manifest with page offsets."""
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)
    manifest: list[dict] = []
    for idx, start in enumerate(range(0, total, shard_pages)):
        end = min(start + shard_pages, total)
        writer = PdfWriter()
        for p in range(start, end):
            writer.add_page(reader.pages[p])
        name = f"shard_{idx:03d}.pdf"
        with open(out_dir / name, "wb") as fh:
            writer.write(fh)
        manifest.append(
            {
                "shard": name,
                "page_offset": start,      # 0-based offset of this shard's first page
                "page_start": start + 1,   # 1-based, for humans
                "page_end": end,
                "n_pages": end - start,
            }
        )
    assert sum(s["n_pages"] for s in manifest) == total, "shard page counts must sum to the document"
    return manifest


def upload_shards(manifest: list[dict], local_dir: Path, in_bucket: str, prefix: str) -> None:
    project = env("GCP_PROJECT_ID")
    client = storage.Client(project=project, credentials=credentials(project))
    bucket = client.bucket(in_bucket)
    for shard in manifest:
        blob = bucket.blob(f"{prefix}/{shard['shard']}")
        blob.upload_from_filename(str(local_dir / shard["shard"]))
        shard["gcs_uri"] = f"gs://{in_bucket}/{prefix}/{shard['shard']}"


# --------------------------------------------------------------------------- #
# batch process
# --------------------------------------------------------------------------- #
def _split_uri(uri: str) -> tuple[str, str]:
    m = re.match(r"gs://([^/]+)/?(.*)", uri)
    if not m:
        raise ValueError(f"not a gs:// uri: {uri}")
    return m.group(1), m.group(2)


def batch_process(manifest: list[dict], output_uri: str) -> dict[str, str]:
    """Run one batch request over every shard. Returns {shard_gcs_uri: output_prefix}."""
    location = env("GCP_LOCATION", "eu")
    project = env("GCP_PROJECT_ID")
    client = documentai.DocumentProcessorServiceClient(
        credentials=credentials(project),
        client_options=ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com"),
    )
    name = client.processor_path(project, location, env("DOCAI_PROCESSOR_ID"))

    request = documentai.BatchProcessRequest(
        name=name,
        input_documents=documentai.BatchDocumentsInputConfig(
            gcs_documents=documentai.GcsDocuments(
                documents=[
                    documentai.GcsDocument(gcs_uri=s["gcs_uri"], mime_type="application/pdf")
                    for s in manifest
                ]
            )
        ),
        document_output_config=documentai.DocumentOutputConfig(
            gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(gcs_uri=output_uri)
        ),
        # Layout/paragraph detection is what gives us blocks with bounding boxes;
        # without bboxes the boilerplate detector in stage 1.2 cannot run.
        process_options=documentai.ProcessOptions(
            ocr_config=documentai.OcrConfig(
                enable_symbol=False,
                enable_image_quality_scores=False,
            )
        ),
    )

    print(f"submitting batch over {len(manifest)} shards -> {output_uri}")
    operation = client.batch_process_documents(request)
    operation.result(timeout=3600)
    metadata = documentai.BatchProcessMetadata(operation.metadata)
    if metadata.state != documentai.BatchProcessMetadata.State.SUCCEEDED:
        raise RuntimeError(f"batch failed: {metadata.state} {metadata.state_message}")

    mapping: dict[str, str] = {}
    for status in metadata.individual_process_statuses:
        if status.status.code != 0:
            raise RuntimeError(f"shard failed: {status.input_gcs_source} {status.status.message}")
        mapping[status.input_gcs_source] = status.output_gcs_destination
    return mapping


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
def _anchor_text(doc_text: str, anchor) -> str:
    if not anchor or not anchor.text_segments:
        return ""
    return "".join(
        doc_text[int(seg.start_index) : int(seg.end_index)] for seg in anchor.text_segments
    )


def _bbox(layout, page_w: float, page_h: float) -> list[float] | None:
    poly = layout.bounding_poly
    verts = list(poly.normalized_vertices) or list(poly.vertices)
    if not verts:
        return None
    normalised = bool(poly.normalized_vertices)
    xs = [v.x * page_w if normalised else v.x for v in verts]
    ys = [v.y * page_h if normalised else v.y for v in verts]
    return [min(xs), min(ys), max(xs), max(ys)]


def normalise_document(doc: documentai.Document, page_offset: int) -> list[dict]:
    """One record per page, blocks carrying bbox + y_top in page points."""
    out: list[dict] = []
    for page in doc.pages:
        page_w = page.dimension.width or 595.0
        page_h = page.dimension.height or 842.0
        units = list(page.paragraphs) or list(page.blocks)
        unit_type = "paragraph" if page.paragraphs else "block"
        blocks = []
        for unit in units:
            text = _anchor_text(doc.text, unit.layout.text_anchor)
            box = _bbox(unit.layout, page_w, page_h)
            if not text.strip():
                continue
            blocks.append(
                {
                    "text": text,
                    "bbox": box,
                    "type": unit_type,
                    "y_top": round(box[1], 2) if box else None,
                    "x_left": round(box[0], 2) if box else None,
                    "x_right": round(box[2], 2) if box else None,
                }
            )
        blocks.sort(key=lambda b: (b["y_top"] if b["y_top"] is not None else 0,
                                   b["x_left"] if b["x_left"] is not None else 0))
        out.append(
            {
                "page": page_offset + int(page.page_number),  # global 1-based page
                "page_width": page_w,
                "page_height": page_h,
                "text": _anchor_text(doc.text, page.layout.text_anchor),
                "blocks": blocks,
            }
        )
    return out


def download_and_merge(manifest: list[dict], mapping: dict[str, str]) -> list[dict]:
    project = env("GCP_PROJECT_ID")
    client = storage.Client(project=project, credentials=credentials(project))
    pages: list[dict] = []
    for shard in manifest:
        out_prefix = mapping[shard["gcs_uri"]]
        bucket_name, prefix = _split_uri(out_prefix)
        bucket = client.bucket(bucket_name)
        blobs = sorted(
            (b for b in client.list_blobs(bucket_name, prefix=prefix) if b.name.endswith(".json")),
            key=lambda b: b.name,
        )
        if not blobs:
            raise RuntimeError(f"no output for shard {shard['shard']} at {out_prefix}")
        for blob in blobs:
            doc = documentai.Document.from_json(
                blob.download_as_bytes(), ignore_unknown_fields=True
            )
            pages.extend(normalise_document(doc, shard["page_offset"]))
    pages.sort(key=lambda p: p["page"])
    return pages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-batch", action="store_true",
                    help="reuse the mapping from a previous run (data/interim/docai_manifest.json)")
    args = ap.parse_args()

    pdf = ROOT / SETTINGS["paths"]["raw_pdf"]
    shard_dir = ROOT / "data" / "interim" / "shards"
    manifest_path = ROOT / "data" / "interim" / "docai_manifest.json"

    in_bucket, _ = _split_uri(env("GCS_INPUT_URI"))
    output_uri = env("GCS_OUTPUT_URI")

    if args.skip_batch and manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        manifest, mapping = saved["manifest"], saved["mapping"]
    else:
        manifest = shard_pdf(pdf, shard_dir)
        print(f"sharded {pdf.name} into {len(manifest)} shards")
        upload_shards(manifest, shard_dir, in_bucket, "shards")
        mapping = batch_process(manifest, output_uri)
        manifest_path.write_text(json.dumps({"manifest": manifest, "mapping": mapping}, indent=2))

    pages = download_and_merge(manifest, mapping)

    expected = SETTINGS["document"]["expected_pages"]
    seen = [p["page"] for p in pages]
    assert seen == list(range(1, len(seen) + 1)), "page numbering must be contiguous from 1"
    n = write_jsonl(path("raw_pages"), pages)

    no_bbox = sum(1 for p in pages for b in p["blocks"] if not b["bbox"])
    report(
        "docai_report.json",
        {
            "pages": n,
            "expected_pages": expected,
            "pages_match": n == expected,
            "shards": len(manifest),
            "blocks": sum(len(p["blocks"]) for p in pages),
            "blocks_without_bbox": no_bbox,
            "chars": sum(len(p["text"]) for p in pages),
        },
    )
    print(f"wrote {n} pages (expected {expected}); blocks without bbox: {no_bbox}")


if __name__ == "__main__":
    main()
