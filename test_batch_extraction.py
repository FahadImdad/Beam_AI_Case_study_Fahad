import os
import io
import json
import re
from typing import Tuple, Dict, Any, Optional

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import AzureOpenAI

# ---------- Optional, higher-fidelity PDF text extraction (prompt-only logic; no data hardcoding)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
try:
    # Some envs expose pdfminer like this
    from pdfminer_high_level import extract_text as pdfminer_extract_text  # type: ignore
except Exception:
    try:
        # Fallback to pdfminer.six canonical import path used in many envs
        from pdfminer.high_level import extract_text as pdfminer_extract_text  # type: ignore
    except Exception:
        pdfminer_extract_text = None


# =========================
# Paths (update here only)
# =========================
PATH_ROOT = r"C:\\Users\\fahad.imdad\\Documents\\Beam_AI_Case_Study_Fahad"
PATH_EXPECTED = os.path.join(PATH_ROOT, "Expected Output - EcomCorp Order Test Dataset.csv")
PATH_RECEIPTS = os.path.join(PATH_ROOT, "Sales Receipt")
PATH_SALES_EMAILS = os.path.join(PATH_ROOT, "Sales Email - EcomCorp Order - Test Dataset.csv")

# TXT dump of first 10 receipts (open in Notepad)
OUT_ALL_RECEIPTS_TXT = os.path.join(PATH_ROOT, "all_receipts_text.txt")


# =========================
# Azure OpenAI client
# =========================
load_dotenv()
azure_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
if not all([azure_key, azure_api_version, azure_endpoint, azure_deployment]):
    raise RuntimeError(
        "Missing Azure OpenAI env vars. Set AZURE_OPENAI_API_KEY (or AZURE_OPENAI_KEY), "
        "AZURE_OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT."
    )
client = AzureOpenAI(api_key=azure_key, api_version=azure_api_version, azure_endpoint=azure_endpoint)


# =========================
# System prompt (kept simple for exact matching)
# =========================
SYSTEM_PROMPT = """
You are a strict information extractor. 
Your task is to return ONE JSON object that exactly matches the provided SCHEMA. 
Do not output anything else. If data is missing or uncertain → return null.

----------------------------
GENERAL RULES
----------------------------
- Return ONLY one JSON object. No prose, no explanations, no extra keys.
- "Normalize" all receipts":
  • Replace "&" with "and" for all receipts.
  • Trim spaces
  • Preserve German diacritics (ä, ö, ü, ß).
- Dates must be in format: YYYY.MM.DD
- Integers must be plain integers (no quotes).
- Never hallucinate values.
- Always give priority to information from the receipt over the email if both are available.

----------------------------
DELIVERY ADDRESS RULES
----------------------------
- Always select the **customer’s delivery/buyer address**, NOT the supplier/vendor masthead.
- Ignore addresses if they contain:
  • Supplier company names (e.g., OttA GmbH, Anhalt Tools, etc.)
  • Known supplier HQ postal codes (e.g., 4718x Willo).
- Prefer addresses explicitly labelled: "Lieferadresse", "Delivery address", "Versandadresse", "Ship to".
- If no explicit label exists:
  • Select the address block closest to the buyer/company information near the order header.
  • Do NOT take addresses from the bottom masthead or repeated headers/footers.
- Example:
  ✅ "Hofbauer Engineering GmbH - Industriestr. 9 - 84326 Falkenberg" → delivery
  ❌ "OttA GmbH, Hans-Martin-Schleyer-Str. 15b, 47186 Willo" → supplier masthead

----------------------------
BUYER RULES
----------------------------
- Use buyer details from the EMAIL header/footer if present.
- Otherwise, use the top section of the RECEIPT labelled as buyer/customer.
- Never select supplier/vendor information as buyer.
- Buyer name should be on top of the RECEIPT.

----------------------------
Priority:
1) From EMAIL header/footer/signature (sender org/person/email).
2) Else from RECEIPT block clearly labelled Customer/Kunde/Billing/Buyer and located near the order header.

Do NOT take supplier/vendor info as buyer.

Disambiguation:
- Ignore salutations to supplier staff (e.g., “Sehr geehrte Frau …”) — those are recipients, not the buyer.
- If multiple names appear (e.g., “Sachbearbeiter/Ansprechpartner”), choose the name associated with the buyer organization, not the supplier.
- If the RECEIPT only shows initials (e.g., “D. Lehmann”) but the EMAIL provides a full name for the same person (e.g., “David Lehmann”), use the full name from EMAIL.
- Buyer email: take from EMAIL if present; otherwise, use a clear buyer-domain address on the RECEIPT (avoid generic supplier domains).


----------------------------
PRODUCT RULES
----------------------------
- Extract each row in the order table.
- product_position: from "Pos" or row index if missing.
- product_quantity: from "Menge"/"Qty" (integer, ignore units like Stk/pcs).
- product_article_code:
  • Use the value under/next to: "Ihre Materialnummer", "Artikelnummer", "Article No", "Item No".
      - choose "Ihre Materialnummer" if it exists.(Always do reasoniung and choose correct product article code not supplier article code).
      - Product article codes must never contain spaces or hyphens.
        If a code appears with a prefix letter and separator (e.g., "F- P45233180", "X 396655", "X-39988"), then:
       Ignore the alphabetic prefix (F, X, etc.) if there exists any hyphen or space after it.Otherwise keep the prefix.
       -Keep only the pure alphanumeric code (e.g., "P45233180", "396655", "39988").
      - Choose the supplier’s actual article/material number (often the alphanumeric code, e.g., X920381742).
      - Do NOT choose internal/customer reference numbers (e.g., MB00022519) unless no other valid code exists.
      - Dont choose supplier article no.
      - Product article code have both alphanumeric and numeric ones.Analyze both and choose the correct one."
      - If a row does not contain a product_article_code, exclude that row entirely.
      - Do not output product_position or product_quantity for excluded rows.
      - Only include rows where a product_article_code is present.
  • Do NOT use order numbers, invoice numbers, postal codes, phone numbers, etc.

----------------------------
OUTPUT CONTRACT
----------------------------
- Must exactly match the JSON SCHEMA keys and types.
- Include all products found (row by row).
- If multiple candidates exist, apply the rules above deterministically.
- If still unclear, return null for the field.
"""




# =========================
# JSON Schema (single source of truth, reused for validation)
# =========================
SCHEMA_JSON: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "buyer": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "buyer_company_name": {"type": ["string", "null"]},
                "buyer_person_name": {"type": ["string", "null"]},
                "buyer_email_address": {"type": ["string", "null"]},
            },
            "required": ["buyer_company_name", "buyer_person_name", "buyer_email_address"],
        },
        "order": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "order_number": {"type": ["string", "null"]},
                "order_date": {
                    "anyOf": [
                        {"type": "string", "pattern": r"^\d{4}\.\d{2}\.\d{2}$"},
                        {"type": "null"}
                    ]
                },
                "delivery_address_street": {"type": ["string", "null"]},
                "delivery_address_city": {"type": ["string", "null"]},
                "delivery_address_postal_code": {"type": ["string", "null"]},
            },
            "required": [
                "order_number",
                "order_date",
                "delivery_address_street",
                "delivery_address_city",
                "delivery_address_postal_code",
            ],
        },
        "products": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "product_position": {"type": "integer"},
                    "product_article_code": {"type": ["string", "null"]},
                    "product_quantity": {"type": ["integer", "null"]},
                },
                "required": ["product_position", "product_article_code", "product_quantity"],
            },
        },
    },
    "required": ["buyer", "order", "products"],
}

# Optional: show the schema to the model verbatim in the user message (helps exact matching)
SCHEMA_TEXT = """
SCHEMA (must match exactly)
{
  "buyer": {
    "buyer_company_name": "<string or null>",
    "buyer_person_name": "<string>",
    "buyer_email_address": "<string or null>"
  },
  "order": {
    "order_number": "<string or null>",
    "order_date": "<YYYY.MM.DD or null>",
    "delivery_address_street": "<string or null>",
    "delivery_address_city": "<string or null>",
    "delivery_address_postal_code": "<string or null>"
  },
  "products": [
    {
      "product_position": "<integer>",
      "product_article_code": "<string or null>",
      "product_quantity": "<integer or null>"
    }
  ]
}

OUTPUT
Return ONLY the JSON object matching SCHEMA. No extra text.
"""

# =========================
# Optional tool-call (“trust call”) extractor toggle
# =========================
USE_TOOLCALL_EXTRACTOR = False  # Set True to force tool-call mode

EXTRACTOR_TOOL = [{
    "type": "function",
    "function": {
        "name": "emit_order_json",
        "description": "Emit the extracted order JSON exactly once.",
        "parameters": SCHEMA_JSON,
        "strict": True
    }
}]


# =========================
# PDF text extraction (layout-aware when possible; still generic)
# =========================
def extract_text_from_pdf(pdf_path: str) -> str:
    # Try PyMuPDF (layout grouping keeps diacritics reliably)
    if fitz is not None:
        try:
            doc = fitz.open(pdf_path)
            parts = []
            for page in doc:
                blocks = page.get_text("blocks")
                # sort by y, then x
                blocks = sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))
                for b in blocks:
                    txt = (b[4] or "").strip()
                    if txt:
                        parts.append(txt)
            doc.close()
            if parts:
                return "\n".join(parts)
        except Exception:
            pass
    # Fallback to pdfminer if available
    if pdfminer_extract_text is not None:
        try:
            return pdfminer_extract_text(pdf_path) or ""
        except Exception:
            pass
    # Last resort
    return ""


# =========================
# Prompt assembly
# =========================
def make_extraction_messages(email_text: str, receipt_text: str):
    user_block = f"""USER INPUT (pass both blocks exactly):
EMAIL:
<<<{email_text or ""}>>>

RECEIPT:
<<<{receipt_text or ""}>>>

{SCHEMA_TEXT}
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_block},
    ]


# =========================
# Helpers for Responses API output
# =========================
def _safe_output_text(resp) -> str:
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text

    out_items = getattr(resp, "output", None)
    if isinstance(out_items, list):
        buf = []
        for item in out_items:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for part in content:
                    t = getattr(part, "text", None)
                    if isinstance(t, str):
                        buf.append(t)
        if buf:
            return "".join(buf)
    return ""


def _find_tool_call_arguments(resp) -> Optional[Dict[str, Any]]:
    out_items = getattr(resp, "output", None)
    if isinstance(out_items, list):
        for item in out_items:
            if getattr(item, "type", None) in ("tool_call", "function_call"):
                if getattr(item, "name", "") == "emit_order_json":
                    args = getattr(item, "arguments", None)
                    if isinstance(args, dict):
                        return args
                    if isinstance(args, str):
                        try:
                            return json.loads(args)
                        except Exception:
                            pass
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for part in content:
                    if getattr(part, "type", None) in ("tool_call", "function_call"):
                        if getattr(part, "name", "") == "emit_order_json":
                            args = getattr(part, "arguments", None)
                            if isinstance(args, dict):
                                return args
                            if isinstance(args, str):
                                try:
                                    return json.loads(args)
                                except Exception:
                                    pass
    return None


# =========================
# LLM call with strict schema (and optional tool-call extractor)
# =========================
def call_llm(messages: list) -> Tuple[str, Optional[dict]]:
    if USE_TOOLCALL_EXTRACTOR:
        try:
            resp = client.responses.create(
                model=azure_deployment,
                input=messages,
                temperature=0,
                max_output_tokens=1500,
                tools=EXTRACTOR_TOOL,
                tool_choice={"type": "function", "function": {"name": "emit_order_json"}},
            )
            tool_args = _find_tool_call_arguments(resp)
            if tool_args is None:
                return "No tool_call with emit_order_json was returned.", None
            return json.dumps(tool_args, ensure_ascii=False), tool_args
        except Exception:
            pass

    try:
        resp = client.responses.create(
            model=azure_deployment,
            input=messages,
            temperature=0,
            max_output_tokens=1500,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "order_extraction",
                    "schema": SCHEMA_JSON,
                    "strict": True,
                }
            },
        )
        text = _safe_output_text(resp) or ""
        candidate = text.strip()
        data = json.loads(candidate)
        return candidate, data
    except Exception as e_schema:
        try:
            resp = client.responses.create(
                model=azure_deployment,
                input=messages,
                temperature=0,
                max_output_tokens=1500,
            )
            text = _safe_output_text(resp) or ""
        except Exception as e2:
            return f"LLM call failed: {e_schema} | Fallback failed: {e2}", None

        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines[0].startswith("```") and lines[-1].startswith("```"):
                stripped = "\n".join(lines[1:-1])

        candidate = stripped
        if not candidate.startswith("{"):
            m = re.search(r"\{[\s\S]*\}\s*$", stripped)
            if m:
                candidate = m.group(0)
        try:
            data = json.loads(candidate)
            return candidate, data
        except json.JSONDecodeError:
            m2 = re.search(r"\{[\s\S]*\}", stripped)
            if m2:
                try:
                    data = json.loads(m2.group(0))
                    return m2.group(0), data
                except Exception:
                    pass
            return f"Non-JSON or invalid JSON returned:\n{text}", None


# =========================
# Evaluation helpers (unchanged)
# =========================
def flatten_paths(d, base=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from flatten_paths(v, f"{base}.{k}" if base else k)
    elif isinstance(d, list):
        for idx, v in enumerate(d, start=1):
            yield from flatten_paths(v, f"{base}[{idx}]")
    else:
        yield base, d


def compare_json(expected: dict, got: dict):
    exp_map = dict(flatten_paths(expected))
    got_map = dict(flatten_paths(got))
    keys = sorted(set(exp_map) | set(got_map))
    rows, match_count, total = [], 0, 0
    for k in keys:
        e, g = exp_map.get(k, None), got_map.get(k, None)
        ok = e == g
        rows.append((k, e, g, ok))
        total += 1
        if ok:
            match_count += 1
    acc = (match_count / total * 100.0) if total else 0.0
    return acc, rows


# =========================
# Build maps from CSVs (unchanged except diacritics fixes)
# =========================
def build_expected_map_from_csv(path: str) -> dict:
    df_exp = pd.read_csv(path, encoding="cp1252")
    expected_map: Dict[int, dict] = {}

    def fix_text(s):
        if pd.isna(s):
            return None
        s = str(s)
        return (s
                .replace("Ã¼", "ü")
                .replace("Ã„", "Ä")
                .replace("Ã¶", "ö")
                .replace("ÃŸ", "ß")
                .replace("Ã–", "Ö")
                .replace("Ã¤", "ä"))

    for idx, row in df_exp.iterrows():
        i = int(row["Index"]) if not pd.isna(row["Index"]) else None
        if i is None:
            continue
        entry = expected_map.get(i)
        if entry is None:
            entry = {
                "buyer": {
                    "buyer_company_name": fix_text(row.get("buyer_company_name")),
                    "buyer_person_name": fix_text(row.get("buyer_person_name")),
                    "buyer_email_address": fix_text(row.get("buyer_email_address")),
                },
                "order": {
                    "order_number": fix_text(row.get("order_number")),
                    "order_date": None,
                    "delivery_address_street": fix_text(row.get("delivery_address_street")),
                    "delivery_address_city": fix_text(row.get("delivery_address_city")),
                    "delivery_address_postal_code": fix_text(row.get("delivery_address_postal_code")),
                },
                "products": [],
            }
            date_val = fix_text(row.get("order_date"))
            if date_val:
                m = re.match(r"(\d{2})[.](\d{2})[.](\d{4})", date_val)
                if m:
                    entry["order"]["order_date"] = f"{m.group(3)}.{m.group(2)}.{m.group(1)}"
                else:
                    entry["order"]["order_date"] = date_val
            expected_map[i] = entry
        pos = row.get("product_position")
        code = fix_text(row.get("product_article_code"))
        qty = row.get("product_quantity")
        if not pd.isna(pos) or not pd.isna(code) or not pd.isna(qty):
            try:
                pos_int = int(pos) if not pd.isna(pos) else None
            except Exception:
                pos_int = None
            try:
                qty_int = int(qty) if not pd.isna(qty) else None
            except Exception:
                qty_int = None
            expected_map[i]["products"].append({
                "product_position": pos_int,
                "product_article_code": code if code else None,
                "product_quantity": qty_int,
            })
    for i, entry in expected_map.items():
        entry["products"].sort(key=lambda p: (p["product_position"] is None, p["product_position"]))
    return expected_map


def build_email_map_from_csv(path: str) -> dict:
    df_em = pd.read_csv(path, encoding="cp1252")
    emails_series = df_em.iloc[:, 0]
    emails = [str(v) for v in emails_series if isinstance(v, str) and v.strip()
              and not v.strip().lower().startswith('sales email')]
    email_map = {i + 1: emails[i] for i in range(min(len(emails), 1000))}
    return email_map


# =========================
# Main
# =========================
def main():
    expected_map = build_expected_map_from_csv(PATH_EXPECTED)
    email_map = build_email_map_from_csv(PATH_SALES_EMAILS)
    indices = sorted(expected_map.keys())
    results = []

    # Collect first 10 receipts' raw text for a single Notepad file
    txt_chunks = []

    for n, pdf_idx in enumerate(tqdm(indices), start=1):
        if n > 10:
            break  # only first 10 PDFs as requested

        pdf_path = os.path.join(PATH_RECEIPTS, f"{pdf_idx}.pdf")
        if not os.path.exists(pdf_path):
            print(f"PDF not found: {pdf_path}")
            # still put a placeholder so you know it was missing
            txt_chunks.append(f"===== Row {pdf_idx} ({pdf_path}) =====\n(PDF not found)\n")
            continue

        email_text = email_map.get(pdf_idx, "")
        receipt_text = extract_text_from_pdf(pdf_path)

        # --- Append to TXT dump (receipt text only) ---
        safe_text = receipt_text if receipt_text.strip() else "(no text extracted)"
        txt_chunks.append(f"===== Row {pdf_idx} ({pdf_path}) =====\n{safe_text}\n")

        # ----- The rest of your evaluation pipeline -----
        messages = make_extraction_messages(email_text, receipt_text)
        raw_text, json_data = call_llm(messages)
        if json_data is None:
            print("RAW AI (unparsed):")
            print(raw_text)
        expected_json = expected_map.get(pdf_idx)

        if json_data is not None and expected_json is not None:
            acc, rows_cmp = compare_json(expected_json, json_data)
        else:
            acc, rows_cmp = 0.0, []

        diffs = [(k, e, g) for (k, e, g, ok) in rows_cmp if not ok]

        results.append({
            "row": pdf_idx,
            "accuracy": acc,
            "ai_output": json_data,
            "expected": expected_json,
            "raw_ai": raw_text,
            "diffs": diffs,
        })

        print(f"\nRow {pdf_idx}:")
        print("AI Output:")
        print(json.dumps(json_data, ensure_ascii=False, indent=2))
        print("Expected Output:")
        print(json.dumps(expected_json, ensure_ascii=False, indent=2))
        print(f"Accuracy: {acc:.2f}%")

        if diffs:
            print("Differences (path → expected vs got):")
            for k, e, g in diffs:
                print(f"- {k}\n    expected: {e}\n    got:      {g}")
        else:
            print("Differences: None ✅")

    # ---------- Write the Notepad-friendly TXT ----------
    try:
        with open(OUT_ALL_RECEIPTS_TXT, "w", encoding="utf-8", errors="ignore") as f:
            f.write("\n".join(txt_chunks))
        print(f"\nWrote first 10 receipts' text to: {OUT_ALL_RECEIPTS_TXT}")
    except Exception as e:
        print(f"Failed to write TXT dump: {e}")

    # ---------- Keep your summary artifacts ----------
    out_rows = [{"row": r["row"], "accuracy": r["accuracy"]} for r in results]
    pd.DataFrame(out_rows).to_csv(os.path.join(PATH_ROOT, "batch_extraction_results.csv"), index=False)

    side_by_side_rows = []
    for r in results:
        side_by_side_rows.append({
            "row": r["row"],
            "accuracy": r["accuracy"],
            "expected_output": json.dumps(r["expected"], ensure_ascii=False, indent=2) if r["expected"] is not None else None,
            "ai_output": json.dumps(r["ai_output"], ensure_ascii=False, indent=2) if r["ai_output"] is not None else None,
        })
    side_by_side_df = pd.DataFrame(side_by_side_rows, columns=["row", "accuracy", "expected_output", "ai_output"])
    side_by_side_df.to_excel(os.path.join(PATH_ROOT, "batch_extraction_comparison.xlsx"), index=False)

    print("\nSummary:")
    acc_values = []
    for r in results:
        print(f"Row {r['row']}: Accuracy={r['accuracy']:.2f}%")
        acc_values.append(r["accuracy"])
    if acc_values:
        avg_acc = sum(acc_values) / len(acc_values)
        print(f"Average Accuracy: {avg_acc:.2f}%")
    print("\nWrote side-by-side comparison to batch_extraction_comparison.xlsx and summary CSV to batch_extraction_results.csv")


if __name__ == "__main__":
    main()
