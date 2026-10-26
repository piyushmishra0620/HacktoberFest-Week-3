"""Turn any uploaded file into text + images that a Gemma model can read."""
import io, os, subprocess, tempfile, glob
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

MAX_PAGES_AS_IMAGES = int(os.getenv("MAX_PAGES_AS_IMAGES", 8))
MAX_EMBEDDED_IMAGES = int(os.getenv("MAX_EMBEDDED_IMAGES", 6))
MAX_SHEET_ROWS = int(os.getenv("MAX_SHEET_ROWS", 60))


@dataclass
class Extracted:
    text: str = ""
    images: list = field(default_factory=list)   # raw image bytes (PNG/JPEG)
    notes: list = field(default_factory=list)    # warnings shown to the user


def _shrink(data: bytes, max_side: int = 1024) -> bytes:
    try:
        im = Image.open(io.BytesIO(data)).convert("RGB")
        im.thumbnail((max_side, max_side))
        out = io.BytesIO()
        im.save(out, "JPEG", quality=85)
        return out.getvalue()
    except Exception:
        return b""


# ---------- documents ----------
def _pdf(data: bytes) -> Extracted:
    import fitz
    ex, parts = Extracted(), []
    doc = fitz.open(stream=data, filetype="pdf")
    scanned = 0
    for i, page in enumerate(doc, 1):
        txt = page.get_text().strip()
        if len(txt) >= 40:
            parts.append(f"--- Page {i} ---\n{txt}")
        elif scanned < MAX_PAGES_AS_IMAGES:
            scanned += 1
            pix = page.get_pixmap(dpi=110)
            ex.images.append(pix.tobytes("png"))
            parts.append(f"--- Page {i} --- (image-only page, see visuals)")
    if scanned:
        ex.notes.append(f"{scanned} image-only page(s) were read visually (OCR-style).")
    # embedded figures on text pages
    n = 0
    for page in doc:
        for img in page.get_images(full=True):
            if n >= MAX_EMBEDDED_IMAGES:
                break
            try:
                base = doc.extract_image(img[0])
                if base["width"] >= 200 and base["height"] >= 200:
                    ex.images.append(base["image"]); n += 1
            except Exception:
                pass
    ex.text = "\n\n".join(parts)
    return ex


def _docx(data: bytes) -> Extracted:
    import docx
    d = docx.Document(io.BytesIO(data))
    ex, parts = Extracted(), []
    for p in d.paragraphs:
        if p.text.strip():
            style = (p.style.name or "") if p.style is not None else ""
            parts.append(("# " if style.startswith("Heading") else "") + p.text)
    for ti, t in enumerate(d.tables, 1):
        rows = [" | ".join(c.text.strip() for c in r.cells) for r in t.rows]
        parts.append(f"[Table {ti}]\n" + "\n".join(rows))
    for rel in d.part.rels.values():
        if "image" in rel.reltype and len(ex.images) < MAX_EMBEDDED_IMAGES:
            ex.images.append(rel.target_part.blob)
    ex.text = "\n\n".join(parts)
    return ex


def _pptx(data: bytes) -> Extracted:
    from pptx import Presentation
    prs = Presentation(io.BytesIO(data))
    ex, parts = Extracted(), []
    for i, slide in enumerate(prs.slides, 1):
        lines = []
        for sh in slide.shapes:
            if sh.has_text_frame and sh.text_frame.text.strip():
                lines.append(sh.text_frame.text.strip())
            if sh.shape_type == 13 and len(ex.images) < MAX_EMBEDDED_IMAGES:  # picture
                ex.images.append(sh.image.blob)
        if slide.has_notes_slide:
            nt = slide.notes_slide.notes_text_frame.text.strip()
            if nt:
                lines.append(f"(Speaker notes: {nt})")
        parts.append(f"--- Slide {i} ---\n" + "\n".join(lines))
    ex.text = "\n\n".join(parts)
    return ex


def _sheet(data: bytes, name: str) -> Extracted:
    import pandas as pd
    ext = Path(name).suffix.lower()
    if ext in (".csv", ".tsv"):
        sheets = {"data": pd.read_csv(io.BytesIO(data), sep=None, engine="python")}
    else:
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
    parts = []
    for sname, df in sheets.items():
        df = df.dropna(how="all").dropna(axis=1, how="all")
        block = [f"[Sheet: {sname}] {df.shape[0]} rows x {df.shape[1]} columns",
                 "Columns: " + ", ".join(map(str, df.columns))]
        num = df.select_dtypes("number")
        if not num.empty:
            block.append("Numeric stats:\n" + num.describe().round(2).to_string())
        block.append(f"First {min(len(df), MAX_SHEET_ROWS)} rows (CSV):\n"
                     + df.head(MAX_SHEET_ROWS).to_csv(index=False))
        parts.append("\n".join(block))
    ex = Extracted(text="\n\n".join(parts))
    ex.notes.append("Large sheets are sampled (first rows + statistics), not read in full.")
    return ex


def _html(data: bytes) -> Extracted:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(data.decode("utf-8", "ignore"), "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    return Extracted(text=soup.get_text("\n", strip=True))


# ---------- multimedia ----------
def _image(data: bytes) -> Extracted:
    return Extracted(images=[data], text="(Single image upload)")


_whisper = None
def _transcribe(path: str) -> str:
    global _whisper
    from faster_whisper import WhisperModel
    if _whisper is None:
        _whisper = WhisperModel(os.getenv("WHISPER_MODEL", "base"), compute_type="int8")
    segs, _ = _whisper.transcribe(path, vad_filter=True)
    return " ".join(s.text.strip() for s in segs)


def _av(data: bytes, name: str, video: bool) -> Extracted:
    ex = Extracted()
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in" + Path(name).suffix)
        Path(src).write_bytes(data)
        wav = os.path.join(tmp, "a.wav")
        try:
            subprocess.run(["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", wav],
                           check=True, capture_output=True)
            ex.text = "[Transcript]\n" + _transcribe(wav)
        except FileNotFoundError:
            ex.notes.append("ffmpeg not installed - cannot transcribe audio.")
        except ImportError:
            ex.notes.append("faster-whisper not installed - cannot transcribe audio.")
        except subprocess.CalledProcessError:
            ex.notes.append("No audio track could be extracted.")
        if video:
            try:
                subprocess.run(["ffmpeg", "-y", "-i", src, "-vf", "fps=1/15,scale=768:-1",
                                "-frames:v", "6", os.path.join(tmp, "f_%02d.jpg")],
                               check=True, capture_output=True)
                for f in sorted(glob.glob(os.path.join(tmp, "f_*.jpg"))):
                    ex.images.append(Path(f).read_bytes())
            except Exception:
                ex.notes.append("Could not sample video frames.")
    return ex


# ---------- dispatcher ----------
TEXT_EXT = {".txt", ".md", ".json", ".xml", ".log", ".py", ".js", ".ts", ".java", ".c", ".cpp",
            ".go", ".rs", ".sql", ".yaml", ".yml", ".ini", ".rtf", ".tex"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def extract(name: str, data: bytes) -> Extracted:
    ext = Path(name).suffix.lower()
    if ext == ".pdf": ex = _pdf(data)
    elif ext == ".docx": ex = _docx(data)
    elif ext == ".pptx": ex = _pptx(data)
    elif ext in (".xlsx", ".xlsm", ".xls", ".csv", ".tsv"): ex = _sheet(data, name)
    elif ext in (".html", ".htm"): ex = _html(data)
    elif ext in IMG_EXT: ex = _image(data)
    elif ext in AUDIO_EXT: ex = _av(data, name, video=False)
    elif ext in VIDEO_EXT: ex = _av(data, name, video=True)
    elif ext in TEXT_EXT or not ext: ex = Extracted(text=data.decode("utf-8", "ignore"))
    else:
        try:
            ex = Extracted(text=data.decode("utf-8"))
        except UnicodeDecodeError:
            raise ValueError(f"Unsupported file type: {ext}")
    ex.images = [b for b in (_shrink(i) for i in ex.images) if b]
    return ex
