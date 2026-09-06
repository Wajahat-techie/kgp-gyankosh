"""
OCR Ingestion Module for KGP Gyankosh
Handles OCR extraction for scanned government notices, orders, and circulars
using pytesseract and pdf2image with comprehensive error handling and fallback logic.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

logger = logging.getLogger("kgp_gyankosh.ocr")

# Configure Tesseract binary path if provided in environment
tesseract_binary = os.getenv("TESSERACT_CMD", "").strip()
if tesseract_binary and os.path.exists(tesseract_binary):
    pytesseract.pytesseract.tesseract_cmd = tesseract_binary
    logger.info(f"Custom Tesseract binary configured: {tesseract_binary}")


def check_ocr_availability() -> Dict[str, Any]:
    """
    Checks if Tesseract OCR and Poppler binaries are accessible on the host machine.
    Uses non-blocking path and shutil checks to prevent subprocess hangs.
    """
    import shutil
    import subprocess

    status = {
        "tesseract_available": False,
        "poppler_available": False,
        "message": ""
    }
    
    # Check Tesseract binary
    tess_cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")
    is_tess_found = False
    if os.path.isabs(tess_cmd) and os.path.exists(tess_cmd):
        is_tess_found = True
    elif shutil.which(tess_cmd):
        is_tess_found = True
        
    if is_tess_found:
        try:
            res = subprocess.run([tess_cmd, "--version"], capture_output=True, text=True, timeout=2)
            if res.returncode == 0:
                status["tesseract_available"] = True
                status["tesseract_version"] = res.stdout.splitlines()[0] if res.stdout else "active"
        except Exception as e:
            status["tesseract_error"] = str(e)
    else:
        status["tesseract_error"] = f"Binary '{tess_cmd}' not found in PATH or TESSERACT_CMD."

    # Check Poppler
    poppler_path = os.getenv("POPPLER_PATH", "").strip() or None
    try:
        if poppler_path and os.path.exists(poppler_path):
            status["poppler_available"] = True
        elif shutil.which("pdftoppm"):
            status["poppler_available"] = True
        else:
            status["poppler_error"] = "Poppler pdftoppm not found in PATH or POPPLER_PATH."
    except Exception as e:
        status["poppler_error"] = str(e)

    if not status["tesseract_available"]:
        status["message"] = (
            "Tesseract OCR is not installed or not in PATH. "
            "Scanned image OCR will fall back to available text layers. "
            "Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki on Windows."
        )
    else:
        status["message"] = "Tesseract OCR engine is active and ready."
        
    return status


def _ocr_with_rapidocr(image_or_array) -> str:
    """Helper to perform OCR using RapidOCR (pure Python / ONNX)."""
    try:
        from rapidocr_onnxruntime import RapidOCR
        import numpy as np
        ocr = RapidOCR()
        if not isinstance(image_or_array, np.ndarray):
            arr = np.array(image_or_array)
        else:
            arr = image_or_array
        result, _ = ocr(arr)
        if result:
            lines = [line[1] for line in result if line and len(line) > 1 and line[1]]
            return "\n".join(lines).strip()
    except Exception as e:
        logger.warning(f"RapidOCR fallback encountered an issue: {e}")
    return ""


def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image file (.png, .jpg, .jpeg, .tiff, .bmp) using OCR.
    Tries Tesseract first; seamlessly falls back to RapidOCR if Tesseract is missing.
    """
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return ""

    try:
        with Image.open(image_path) as img:
            # 1. Try Tesseract if available
            try:
                img_gray = img.convert("L")
                text = pytesseract.image_to_string(img_gray, lang="eng")
                if text and text.strip():
                    return text.strip()
            except Exception:
                pass

            # 2. Fall back to RapidOCR
            logger.info(f"Using RapidOCR engine for image: {os.path.basename(image_path)}")
            return _ocr_with_rapidocr(img)
    except Exception as exc:
        logger.error(f"Failed OCR on image {image_path}: {exc}", exc_info=True)
        return ""


def extract_text_from_scanned_pdf(
    pdf_path: str,
    poppler_path: Optional[str] = None,
    dpi: int = 200
) -> List[Dict[str, Any]]:
    """
    Converts a scanned PDF without text layer into images per page,
    then performs OCR on each page.
    Checks precomputed OCR cache first; then tries Poppler/Tesseract;
    falls back cleanly to pypdfium2 + RapidOCR if Poppler/Tesseract are not installed.
    """
    import json
    base_name = os.path.basename(pdf_path)
    base_stem = os.path.splitext(base_name)[0]

    # Check if there is a precomputed OCR cache in output/ocr_cache/
    cand_dirs = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "output", "ocr_cache"),
        os.path.join(os.getcwd(), "output", "ocr_cache"),
        os.path.join(os.getcwd(), "kgp-gyankosh", "output", "ocr_cache")
    ]
    for c_dir in cand_dirs:
        for cand_file in [f"{base_stem}_pdf_pages.json", f"{base_name}_pages.json", f"{base_stem}_pages.json"]:
            full_cand = os.path.join(c_dir, cand_file)
            if os.path.exists(full_cand):
                try:
                    with open(full_cand, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    logger.info(f"Loaded {len(data)} cached OCR pages from {full_cand}")
                    return [{"page": int(k), "text": v, "is_ocr": True} for k, v in sorted(data.items(), key=lambda x: int(x[0]))]
                except Exception as ce:
                    logger.warning(f"Error loading OCR cache {full_cand}: {ce}")

    poppler_dir = poppler_path or os.getenv("POPPLER_PATH", "").strip() or None
    page_results: List[Dict[str, Any]] = []

    # 1. Attempt rendering via pdf2image (if Poppler is configured)
    pdf2image_success = False
    try:
        images = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_dir)
        pdf2image_success = True
        for idx, page_img in enumerate(images, start=1):
            try:
                img_gray = page_img.convert("L")
                text = pytesseract.image_to_string(img_gray, lang="eng").strip()
                page_results.append({
                    "page": idx,
                    "text": text,
                    "is_ocr": True
                })
            except Exception:
                # Try RapidOCR for this page
                text = _ocr_with_rapidocr(page_img)
                page_results.append({"page": idx, "text": text, "is_ocr": True})
    except Exception as exc:
        logger.info(f"pdf2image rendering skipped ({exc}). Falling back to pure Python pypdfium2...")

    # 2. Seamless fallback to pypdfium2 + RapidOCR (no system binaries required)
    if not pdf2image_success:
        try:
            import pypdfium2 as pdfium
            logger.info(f"Rendering scanned PDF via pypdfium2 and extracting text via RapidOCR: {os.path.basename(pdf_path)}")
            pdf = pdfium.PdfDocument(pdf_path)
            for idx in range(len(pdf)):
                page = pdf[idx]
                page_img = page.render(scale=2).to_pil()
                text = _ocr_with_rapidocr(page_img)
                page_results.append({
                    "page": idx + 1,
                    "text": text,
                    "is_ocr": True
                })
                logger.info(f"OCR extracted page {idx + 1}/{len(pdf)} ({len(text)} chars)")
        except Exception as p2_err:
            logger.error(f"pypdfium2 + RapidOCR fallback failed for {pdf_path}: {p2_err}", exc_info=True)

    return page_results
