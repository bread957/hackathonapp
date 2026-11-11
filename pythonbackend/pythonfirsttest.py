"""
PDF paper-saver: rewrites a PDF so it prints on fewer sheets of paper.

What it does:
  1. Removes blank pages.
  2. Strips repeated headers/footers and page numbers (they become wrong
     once content is reflowed anyway).
  3. Crops each page down to its actual content (gets rid of big margins).
  4. "Flows" the cropped content onto new pages, one after another, so a page
     with only a line or two of text gets pulled up under the previous page's
     content instead of using a whole sheet. Content is only split between
     lines, never through a line of text, image, or table.

Usage:
    python pythonfirsttest.py input.pdf [output.pdf]
"""

import re
import sys
from collections import Counter

import pymupdf as fitz  # `import fitz` still works but is deprecated

pointsToInches = 72          # 72 PDF points = 1 inch
bigMargin = 10

# Default settings for the output
OUTPUT_MARGIN = 0.5 * pointsToInches   # margin on the new pages (0.5 inch)
GAP_BETWEEN_PAGES = 12                 # vertical space between content from different source pages
CONTENT_PADDING = 2                    # extra room around content so descenders aren't clipped
MIN_SPLIT_HEIGHT = 24                  # don't split off a chunk shorter than this (~2 lines)
HEADER_FOOTER_ZONE = 0.08              # top/bottom 8% of the page is checked for headers/footers

# Rough environmental figures per sheet of office paper (for the website's stats).
# These are commonly cited ballpark estimates, not precise measurements.
GRAMS_PAPER_PER_SHEET = 4.5
GRAMS_CO2_PER_SHEET = 4.6
LITERS_WATER_PER_SHEET = 1.0


def isBlank(page):
    pix = page.get_pixmap(matrix=fitz.Matrix(0.25, 0.25), colorspace=fitz.csGRAY)
    pixels = pix.samples
    isDark = 0
    for p in pixels:
        if p < 240:
            isDark += 1
    return isDark / len(pixels) < 0.001


# ---------------------------------------------------------------------------
# Headers / footers / page numbers
# ---------------------------------------------------------------------------

PAGE_NUMBER_PATTERN = re.compile(r"^(page\s*)?#(\s*(of|/)\s*#)?$")


def _normalize(text):
    """Lowercase, collapse whitespace, replace digits with '#' so
    'Page 3 of 10' and 'Page 4 of 10' look the same."""
    text = re.sub(r"\d+", "#", text.strip().lower())
    text = re.sub(r"[\s\-–—|]+", " ", text).strip()
    return text


def find_header_footer_rects(doc):
    """Return, for every page, a list of rects that are headers, footers or
    page numbers and should be ignored."""
    per_page = []
    counts = Counter()

    for page in doc:
        h = page.rect.height
        items = []
        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text, _, block_type = block
            if block_type != 0 or not text.strip():
                continue
            r = fitz.Rect(x0, y0, x1, y1)
            if r.y1 < h * HEADER_FOOTER_ZONE or r.y0 > h * (1 - HEADER_FOOTER_ZONE):
                items.append((r, _normalize(text)))
        per_page.append(items)
        counts.update({key for _, key in items})  # count each key once per page

    repeat_threshold = max(2, len(doc) * 0.5)
    result = []
    for items in per_page:
        ignore = []
        for r, key in items:
            is_page_number = PAGE_NUMBER_PATTERN.match(key) is not None
            is_repeated = len(doc) > 1 and counts[key] >= repeat_threshold
            if is_page_number or is_repeated:
                ignore.append(r)
        result.append(ignore)
    return result


# ---------------------------------------------------------------------------
# Content detection
# ---------------------------------------------------------------------------

def get_content_boxes(page, ignore_rects=()):
    """Every box on the page that has visible content: text lines, images,
    and drawings (lines, shapes, table borders)."""
    boxes = []

    # Text, line by line (finer than blocks, so we can split between lines)
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        block_rect = fitz.Rect(block["bbox"])
        if any(block_rect.intersects(r) for r in ignore_rects):
            continue
        for line in block["lines"]:
            text = "".join(span["text"] for span in line["spans"])
            if text.strip():
                boxes.append(fitz.Rect(line["bbox"]))

    # Images
    for info in page.get_image_info():
        boxes.append(fitz.Rect(info["bbox"]))

    # Lines, shapes, etc. (skip huge ones, which are usually page backgrounds)
    page_area = page.rect.width * page.rect.height
    for drawing in page.get_drawings():
        r = drawing["rect"]
        if r.width * r.height < 0.95 * page_area:
            boxes.append(fitz.Rect(r))

    # Only keep things actually on the page
    return [b & page.rect for b in boxes if b.intersects(page.rect)]


def get_content_rect(page, ignore_rects=()):
    """Smallest rectangle containing all content on the page (or None)."""
    boxes = get_content_boxes(page, ignore_rects)
    if not boxes:
        return None

    combined = fitz.Rect(boxes[0])
    for box in boxes[1:]:
        combined.include_rect(box)  # grow the rectangle to include this box
    return combined


def find_safe_cuts(boxes):
    """Y positions where the page can be cut horizontally without slicing
    through any text line, image or drawing (i.e. the gaps between rows)."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b.y0)
    cuts = []
    bottom = boxes[0].y1
    for b in boxes[1:]:
        # Line boxes in single-spaced text overlap a little (the box includes
        # line spacing), so a tiny overlap still counts as a gap between rows.
        tolerance = min(3, 0.25 * b.height)
        if b.y0 >= bottom - tolerance:
            cuts.append((bottom + b.y0) / 2)
        bottom = max(bottom, b.y1)
    return cuts


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def optimize_pdf(
    input_path,
    output_path,
    margin=OUTPUT_MARGIN,
    gap=GAP_BETWEEN_PAGES,
    remove_headers_footers=True,
    allow_split=True,
    double_sided=False,
):
    """Write a paper-saving version of input_path to output_path.
    Returns a dict of stats (handy for showing on the website)."""
    src = fitz.open(input_path)
    original_pages = len(src)

    # Work in unrotated coordinates so rotated pages behave like normal ones
    for page in src:
        if page.rotation:
            page.remove_rotation()

    ignore = find_header_footer_rects(src) if remove_headers_footers else [[] for _ in src]

    # 1. Collect the content of every non-blank page
    pieces = []          # (page_number, content_rect, safe_cuts)
    blank_pages = 0
    for page in src:
        rect = get_content_rect(page, ignore[page.number])
        if rect is None or isBlank(page):
            blank_pages += 1
            continue
        boxes = get_content_boxes(page, ignore[page.number])
        rect = (rect + (-CONTENT_PADDING, -CONTENT_PADDING, CONTENT_PADDING, CONTENT_PADDING)) & page.rect
        pieces.append((page.number, rect, find_safe_cuts(boxes)))

    out = fitz.open()
    if not pieces:
        out.new_page(width=src[0].rect.width, height=src[0].rect.height)
        out.save(output_path)
        return _stats(original_pages, 1, blank_pages, double_sided)

    # Output pages use the same size as the original's first page
    W, H = src[0].rect.width, src[0].rect.height
    usable_w = W - 2 * margin
    usable_h = H - 2 * margin

    # Use one shared horizontal crop for every page, so content keeps the same
    # alignment it had originally (centered titles stay centered, etc.)
    left = min(r.x0 for _, r, _ in pieces)
    right = max(r.x1 for _, r, _ in pieces)
    content_w = right - left
    scale = min(1.0, usable_w / content_w)   # only shrink if it's wider than the page
    x_start = margin + (usable_w - content_w * scale) / 2

    # 2. Flow content onto output pages
    current = None
    y = margin  # where the next piece goes on the current output page

    def new_output_page():
        nonlocal current, y
        current = out.new_page(width=W, height=H)
        y = margin

    def place(pno, top, bottom, s=scale):
        nonlocal y
        clip = fitz.Rect(left, top, right, bottom)
        target = fitz.Rect(x_start, y, x_start + content_w * s, y + (bottom - top) * s)
        current.show_pdf_page(target, src, pno, clip=clip)
        y += (bottom - top) * s

    for pno, rect, cuts in pieces:
        top = rect.y0
        if current is None:
            new_output_page()
        elif y > margin:
            y += gap  # space between content from different source pages

        while top < rect.y1:
            remaining = (H - margin - y) / scale     # space left, in source units
            height = rect.y1 - top

            if height <= remaining:                  # the rest fits
                place(pno, top, rect.y1)
                break

            # Doesn't fit: try to split at the lowest safe gap that still fits
            cut = None
            if allow_split:
                fits = [c for c in cuts if top + MIN_SPLIT_HEIGHT <= c <= top + remaining]
                if fits:
                    cut = max(fits)

            if cut is not None:
                place(pno, top, cut)
                top = cut
                new_output_page()
            elif y > margin:
                new_output_page()                    # try again on a fresh page
            else:
                # Fresh page and still no safe place to split: shrink to fit
                place(pno, top, rect.y1, s=min(scale, usable_h / height))
                break

    out.save(output_path, garbage=4, deflate=True)
    return _stats(original_pages, len(out), blank_pages, double_sided)


def _stats(original_pages, new_pages, blank_pages, double_sided):
    sheets = (lambda n: (n + 1) // 2) if double_sided else (lambda n: n)
    sheets_saved = sheets(original_pages) - sheets(new_pages)
    return {
        "original_pages": original_pages,
        "optimized_pages": new_pages,
        "blank_pages_removed": blank_pages,
        "pages_saved": original_pages - new_pages,
        "sheets_saved": sheets_saved,
        "percent_saved": round(100 * (original_pages - new_pages) / original_pages, 1) if original_pages else 0,
        "paper_saved_grams": round(sheets_saved * GRAMS_PAPER_PER_SHEET, 1),
        "co2_saved_grams": round(sheets_saved * GRAMS_CO2_PER_SHEET, 1),
        "water_saved_liters": round(sheets_saved * LITERS_WATER_PER_SHEET, 1),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pythonfirsttest.py input.pdf [output.pdf]")
        sys.exit(1)

    input_pdf = sys.argv[1]
    output_pdf = sys.argv[2] if len(sys.argv) > 2 else input_pdf.rsplit(".", 1)[0] + "_optimized.pdf"

    stats = optimize_pdf(input_pdf, output_pdf)
    print(f"Saved optimized PDF to {output_pdf}")
    for key, value in stats.items():
        print(f"  {key}: {value}")
