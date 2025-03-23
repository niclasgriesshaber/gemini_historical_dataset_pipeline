#!/usr/bin/env python3
"""
Gemini-2.0 PDF -> PNG -> JSON -> CSV Pipeline

This script:
  1) Converts a PDF into per-page PNG images in data/csvs/<pdf_stem>/page_by_page/PNG/.
     (Skips conversion if images already exist.)
  2) Calls Gemini-2.0 for each page image, retrieving JSON output.
     - Automatically retries on any error (including JSON parsing failures).
     - Limits each page's retry attempts to 2 minutes; if no success within that time, it skips the page.
  3) Merges **all** returned JSON data (from the entire run_page_json_dir) into a single CSV 
     named data/csvs/<pdf_stem>/<pdf_stem>.csv.
"""

import os
import sys
import re
import json
import time
import argparse
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Union, Dict, List, Optional

from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image

# Google-GenAI (Gemini-2.0) library
import google.genai as genai
from google.genai import types

###############################################################################
# Project Paths
###############################################################################
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "data" / "csvs"   # Under this, we'll create <pdf_stem> subfolders.
ENV_PATH = PROJECT_ROOT / "config" / ".env"

###############################################################################
# Load Environment Variables
###############################################################################
load_dotenv(dotenv_path=ENV_PATH)
API_KEY = os.getenv("GOOGLE_API_KEY")

# Model constants
MODEL_NAME = "gemini-2.0"           # Short name for folder naming
FULL_MODEL_NAME = "gemini-2.0-flash"
MAX_OUTPUT_TOKENS = 8192
RETRY_LIMIT_SECONDS = 120  # 2 minutes max retry time per page

###############################################################################
# Utility: Time formatting
###############################################################################
def format_duration(seconds: float) -> str:
    """
    Convert a number of seconds into H:MM:SS for clean logging.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

###############################################################################
# Utility: Parse JSON from Gemini-2.0 text response
###############################################################################
def parse_json_str(response_text: str) -> Any:
    """
    Extract code-fenced JSON if present, otherwise fallback to raw text.
    Then parse it as JSON. Raises ValueError if parsing fails.
    """
    fenced_match = re.search(
        r"```(?:json)?\s*([\s\S]*?)\s*```",
        response_text,
        re.IGNORECASE,
    )
    if fenced_match:
        candidate = fenced_match.group(1).strip()
    else:
        # fallback to entire response, removing any backticks
        candidate = response_text.strip().strip("`")

    return json.loads(candidate)

###############################################################################
# Utility: Reorder dictionary keys with page_number at the end
###############################################################################
def reorder_dict_with_page_number(d: Dict[str, Any], page_number: int) -> Dict[str, Any]:
    """
    Return a new dict that includes a 'page_number' key at the end
    and ensures 'additional_information' is last if present.
    """
    special_keys = {"page_number", "additional_information"}
    base_keys = [k for k in d.keys() if k not in special_keys]
    base_keys.sort()

    out = {}
    for k in base_keys:
        out[k] = d[k]

    out["page_number"] = page_number

    if "additional_information" in d:
        out["additional_information"] = d["additional_information"]

    return out

###############################################################################
# Utility: Convert JSON to CSV
###############################################################################
def convert_json_to_csv(json_data: Union[Dict, List], csv_path: Path) -> None:
    """
    Flatten JSON objects/arrays into a CSV at csv_path.
    1) If top-level is a single dict, that's 1 row.
    2) If top-level is a list, each element is a row.
    3) Reorder columns so 'page_number' & 'additional_information' come last.
    """
    import csv

    if isinstance(json_data, dict):
        records = [json_data]
    elif isinstance(json_data, list):
        records = json_data
    else:
        # Fallback: treat as a single row with "value" column
        records = [{"value": str(json_data)}]

    all_keys = set()
    for rec in records:
        if isinstance(rec, dict):
            all_keys.update(rec.keys())

    special_order = ["page_number", "additional_information"]
    base_keys = [k for k in all_keys if k not in special_order]
    base_keys.sort()
    fieldnames = base_keys + [k for k in special_order if k in all_keys]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            if not isinstance(rec, dict):
                # Fallback: convert to dict with "value" column
                row_data = {fn: "" for fn in fieldnames}
                row_data["value"] = str(rec)
                writer.writerow(row_data)
                continue

            row_data = {}
            for fn in fieldnames:
                row_data[fn] = rec.get(fn, "")
            writer.writerow(row_data)

###############################################################################
# Gemini-2.0 API Call with up to 2-minute retry
###############################################################################
def gemini_api_call(
    prompt: str,
    pil_image: Image.Image,
    temperature: float
) -> Optional[dict]:
    """
    Call Gemini-2.0 with the given prompt + image, retry up to 2 minutes if needed.
    Returns a dict:
      {
        "text": <the text response>,
        "usage": <usage metadata object>
      }
    or raises RuntimeError if it fails after 2 minutes.
    """
    client = genai.Client(api_key=API_KEY)
    start_retry = time.time()

    while (time.time() - start_retry) < RETRY_LIMIT_SECONDS:
        tmp_file = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_file = tmp.name
                pil_image.save(tmp_file, "PNG")

            file_upload = client.files.upload(path=tmp_file)

            response = client.models.generate_content(
                model=FULL_MODEL_NAME,
                contents=[
                    types.Part.from_uri(
                        file_uri=file_upload.uri,
                        mime_type=file_upload.mime_type,
                    ),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    response_mime_type="application/json",
                ),
            )

            if not response:
                logging.info("Gemini-2.0 returned an empty response; retrying...")
                continue

            text_candidate = response.text
            if not text_candidate:
                logging.info("Gemini-2.0 returned no text in the response; retrying...")
                continue

            usage = response.usage_metadata
            return {
                "text": text_candidate,
                "usage": usage
            }

        except Exception as e:
            logging.info(f"Gemini-2.0 call failed: {e}. Retrying...")

        finally:
            if tmp_file and os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except:
                    pass

    raise RuntimeError("Gemini-2.0 call did not succeed after 2 minutes - aborting run.")

###############################################################################
# Main
###############################################################################
def main():
    """
    Main entry point for the Gemini-2.0 PDF-to-CSV pipeline.
    """
    try:
        # -------------------------------------------------------------------------
        # Parse arguments
        # -------------------------------------------------------------------------
        parser = argparse.ArgumentParser(description="Gemini-2.0-Flash PDF-to-JSON-to-CSV Pipeline")
        parser.add_argument(
            "--pdfs",
            required=True,
            nargs="+",
            help="One or more PDF files in data/pdfs/, e.g. --pdfs type-1.pdf type-2.pdf"
        )
        parser.add_argument(
            "--temperature",
            type=float,
            default=0.0,
            help="LLM temperature for Gemini-2.0 (default = 0.0)"
        )
        # Changed to default=None so we can detect if it's omitted vs. provided.
        parser.add_argument(
            "--continue_from_page",
            type=int,
            default=None,
            help="Page number to continue from (if omitted, always create a new run)."
        )
        args = parser.parse_args()
        pdf_files = args.pdfs
        temperature = args.temperature
        continue_from_page = args.continue_from_page

        # -------------------------------------------------------------------------
        # Configure logging
        # -------------------------------------------------------------------------
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)]
        )

        logging.info("=== Gemini-2.0 PDF -> PNG -> JSON -> CSV Pipeline ===")
        logging.info(f"Processing {len(pdf_files)} PDFs | Temperature: {temperature}")

        # Process each PDF file
        for pdf_name in pdf_files:
            logging.info(f"\n=== Processing PDF: {pdf_name} ===")
            
            # -------------------------------------------------------------------------
            # Verify the PDF exists
            # -------------------------------------------------------------------------
            pdf_path = DATA_DIR / "pdfs" / pdf_name
            if not pdf_path.is_file():
                logging.error(f"PDF not found at: {pdf_path}")
                continue

            pdf_stem = Path(pdf_name).stem

            # -------------------------------------------------------------------------
            # Load the task prompt for Gemini-2.0
            # -------------------------------------------------------------------------
            prompt_path = PROMPTS_DIR / "prompt.txt"
            if not prompt_path.is_file():
                logging.error(f"Missing prompt file: {prompt_path}")
                continue

            task_prompt = prompt_path.read_text(encoding="utf-8").strip()
            if not task_prompt:
                logging.error(f"Prompt file is empty: {prompt_path}")
                continue

            logging.info(f"Loaded Gemini-2.0 prompt from: {prompt_path}")

            # -------------------------------------------------------------------------
            # Prepare the per-PDF output directory under data/csvs/<pdf_stem>/
            # -------------------------------------------------------------------------
            pdf_out_dir = RESULTS_DIR / pdf_stem
            pdf_out_dir.mkdir(parents=True, exist_ok=True)

            # Subdirectories for PNG and JSON
            png_dir = pdf_out_dir / "page_by_page" / "PNG"
            json_dir = pdf_out_dir / "page_by_page" / "JSON"
            png_dir.mkdir(parents=True, exist_ok=True)
            json_dir.mkdir(parents=True, exist_ok=True)

            # -------------------------------------------------------------------------
            # Check / create PNG pages
            # -------------------------------------------------------------------------
            existing_pngs = list(png_dir.glob("page_*.png"))
            if not existing_pngs:
                logging.info(f"No PNG pages found; converting PDF -> PNG in {png_dir} ...")
                pages = convert_from_path(str(pdf_path))
                for i, pil_img in enumerate(pages, start=1):
                    out_png = png_dir / f"page_{i:04d}.png"
                    pil_img.save(out_png, "PNG")
                logging.info(f"Created {len(pages)} PNG pages in {png_dir}")
            else:
                logging.info(f"PNG folder {png_dir} already has images; skipping PDF->PNG step.")

            # Gather PNG files
            png_files = sorted(png_dir.glob("page_*.png"))
            if not png_files:
                logging.error(f"No PNG pages found in {png_dir}. Exiting.")
                continue

            total_pages = len(png_files)

            # -------------------------------------------------------------------------
            # Accumulators for usage tokens, data for newly processed pages
            # -------------------------------------------------------------------------
            total_prompt_tokens = 0
            total_candidates_tokens = 0
            total_tokens = 0

            # -------------------------------------------------------------------------
            # Start timing the entire pipeline
            # -------------------------------------------------------------------------
            overall_start_time = time.time()

            # -------------------------------------------------------------------------
            # Process each page image from continue_from_page onward
            # -------------------------------------------------------------------------
            start_page = continue_from_page if continue_from_page is not None else 1
            if start_page > 1:
                # Slice the list to skip pages before 'start_page'.
                png_files = png_files[start_page - 1:]

            for idx, png_path in enumerate(png_files, start=start_page):
                logging.info(f"Processing page {idx} of {total_pages}: {png_path.name}")

                # Log image metadata before calling Gemini-2.0
                try:
                    with Image.open(png_path) as pil_image:
                        width, height = pil_image.size
                        dpi_value = pil_image.info.get("dpi", None)
                        if dpi_value and len(dpi_value) == 2:
                            logging.info(
                                f"Image metadata -> width={width}px, height={height}px, dpi={dpi_value}"
                            )
                        else:
                            logging.info(
                                f"Image metadata -> width={width}px, height={height}px, dpi=UNKNOWN"
                            )

                        # Gemini-2.0 call with up to 2-minute retry
                        result = gemini_api_call(
                            prompt=task_prompt,
                            pil_image=pil_image,
                            temperature=temperature
                        )

                except RuntimeError as e:
                    # Construct the continue command with the exact same parameters
                    continue_cmd = f"python {sys.argv[0]} --pdfs {pdf_name}"
                    if temperature != 0.0:  # Only add if not default
                        continue_cmd += f" --temperature {temperature}"
                    continue_cmd += f" --continue_from_page {idx}"
                    
                    logging.error(
                        f"\nFatal error at page {idx} of {pdf_name}: {str(e)}\n"
                        f"To continue processing from this page, run:\n"
                        f"{continue_cmd}"
                    )
                    sys.exit(1)
                except Exception as e:
                    logging.error(f"Failed to open image {png_path}: {e}")
                    continue

                if result is None:
                    logging.error(
                        f"Skipping page {idx} because Gemini-2.0 API did not succeed within 2 minutes."
                    )
                    logging.info("")
                    continue

                response_text = result["text"]
                usage_meta = result["usage"]

                # ---------------------------------------------------------------------
                # Update usage accumulators for this page
                # ---------------------------------------------------------------------
                page_prompt_tokens = usage_meta.prompt_token_count or 0
                page_candidate_tokens = usage_meta.candidates_token_count or 0
                page_total_tokens = usage_meta.total_token_count or 0

                total_prompt_tokens += page_prompt_tokens
                total_candidates_tokens += page_candidate_tokens
                total_tokens += page_total_tokens

                logging.info(
                    f"Gemini-2.0 usage for page {idx}: "
                    f"input={page_prompt_tokens}, candidate={page_candidate_tokens}, total={page_total_tokens}"
                )
                logging.info(
                    f"Accumulated usage so far: input={total_prompt_tokens}, "
                    f"candidate={total_candidates_tokens}, total={total_tokens}"
                )

                # ---------------------------------------------------------------------
                # JSON parsing (retry up to 2 minutes)
                # ---------------------------------------------------------------------
                parse_start_time = time.time()
                parsed = None
                while True:
                    try:
                        parsed = parse_json_str(response_text)
                    except ValueError as ve:
                        if (time.time() - parse_start_time) > RETRY_LIMIT_SECONDS:
                            logging.error(
                                f"Skipping page {idx}: JSON parse still failing after 2 minutes."
                            )
                            break
                        logging.error(f"JSON parse error for page {idx}: {ve}")
                        logging.info(f"Current response text: \n{response_text}\n")
                        logging.error("Retrying Gemini-2.0 call for JSON parse fix...")
                        new_result = gemini_api_call(
                            prompt=task_prompt,
                            pil_image=Image.open(png_path),
                            temperature=temperature
                        )
                        if not new_result:
                            logging.error(
                                f"Could not fix JSON parse for page {idx}, skipping after 2 minutes."
                            )
                            break
                        response_text = new_result["text"]
                        usage_retry = new_result["usage"]

                        # Accumulate usage again
                        retry_ptc = usage_retry.prompt_token_count or 0
                        retry_ctc = usage_retry.candidates_token_count or 0
                        retry_ttc = usage_retry.total_token_count or 0
                        total_prompt_tokens += retry_ptc
                        total_candidates_tokens += retry_ctc
                        total_tokens += retry_ttc

                        logging.info(
                            f"[Retry usage] Additional tokens: input={retry_ptc}, "
                            f"candidate={retry_ctc}, total={retry_ttc} | "
                            f"New accumulated: input={total_prompt_tokens}, "
                            f"candidate={total_candidates_tokens}, total={total_tokens}"
                        )
                    else:
                        break  # parse succeeded

                if not parsed:
                    logging.info("")
                    continue

                # ---------------------------------------------------------------------
                # Save page-level JSON
                # ---------------------------------------------------------------------
                page_json_path = json_dir / f"{png_path.stem}.json"
                with page_json_path.open("w", encoding="utf-8") as jf:
                    json.dump(parsed, jf, indent=2, ensure_ascii=False)

                # ---------------------------------------------------------------------
                # Timing / Estimation
                # ---------------------------------------------------------------------
                elapsed = time.time() - overall_start_time
                pages_done = idx
                pages_left = total_pages - pages_done
                avg_time_per_page = elapsed / pages_done
                estimated_total = avg_time_per_page * total_pages
                estimated_remaining = avg_time_per_page * pages_left

                logging.info(
                    f"Time so far: {format_duration(elapsed)} | "
                    f"Estimated total: {format_duration(estimated_total)} | "
                    f"Estimated remaining: {format_duration(estimated_remaining)}"
                )
                logging.info("")

            # -------------------------------------------------------------------------
            # Now gather *all* JSON files in <pdf_stem>/page_by_page/JSON to create final CSV
            # -------------------------------------------------------------------------
            logging.info("Gathering all JSON files from page_by_page folder to build final CSV...")
            all_page_json_files = sorted(json_dir.glob("page_*.json"))

            merged_data: List[Any] = []

            for fpath in all_page_json_files:
                # page_0001.json => get '0001' => page_number=1
                page_str = fpath.stem.split("_")[1]
                page_num = int(page_str)

                with fpath.open("r", encoding="utf-8") as jf:
                    content = json.load(jf)

                if isinstance(content, list):
                    for obj in content:
                        if isinstance(obj, dict):
                            merged_data.append(reorder_dict_with_page_number(obj, page_num))
                        else:
                            merged_data.append(obj)
                elif isinstance(content, dict):
                    merged_data.append(reorder_dict_with_page_number(content, page_num))
                else:
                    merged_data.append(content)

            # -------------------------------------------------------------------------
            # Add "id" to each record in merged_data
            # -------------------------------------------------------------------------
            for i, record in enumerate(merged_data, start=1):
                if isinstance(record, dict):
                    record["id"] = i
                else:
                    merged_data[i-1] = {"id": i, "value": str(record)}

            # -------------------------------------------------------------------------
            # Convert merged data (all pages) to CSV in data/csvs/<pdf_stem>/<pdf_stem>.csv
            # -------------------------------------------------------------------------
            final_csv_path = pdf_out_dir / f"{pdf_stem}.csv"
            convert_json_to_csv(merged_data, final_csv_path)
            logging.info(f"Final CSV (all pages) saved at: {final_csv_path}")

            # -------------------------------------------------------------------------
            # Final token usage summary
            # -------------------------------------------------------------------------
            total_duration = time.time() - overall_start_time
            logging.info("=== Final Usage Summary ===")
            logging.info(f"Total input tokens used: {total_prompt_tokens}")
            logging.info(f"Total candidate tokens used: {total_candidates_tokens}")
            logging.info(f"Grand total of all tokens used: {total_tokens}")
            logging.info(
                f"Pipeline completed successfully in {format_duration(total_duration)} (H:MM:SS)."
            )
            logging.info("All done!")

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()