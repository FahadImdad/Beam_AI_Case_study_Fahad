import os
import io
import json
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import AzureOpenAI
import re

# ---------- Optional, higher-fidelity PDF text extraction (prompt-only logic; no data hardcoding)
# We keep extraction generic and let the prompt do all decisions.
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:
    pdfminer_extract_text = None

# =========================
# Paths (update here only)
# =========================
PATH_ROOT = r"C:\\Users\\fahad.imdad\\Documents\\Beam_AI_Case_Study_Fahad"
PATH_EXPECTED = os.path.join(PATH_ROOT, "Expected Output - EcomCorp Order Test Dataset.csv")
PATH_RECEIPTS = os.path.join(PATH_ROOT, "Sales Receipt")
PATH_SALES_EMAILS = os.path.join(PATH_ROOT, "Sales Email - EcomCorp Order - Test Dataset.csv")

# =========================
# Azure OpenAI client
# =========================
load_dotenv()
azure_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
if not all([azure_key, azure_api_version, azure_endpoint, azure_deployment]):
    raise RuntimeError("Missing Azure OpenAI env vars. Set AZURE_OPENAI_API_KEY (or AZURE_OPENAI_KEY), AZURE_OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT.")
client = AzureOpenAI(api_key=azure_key, api_version=azure_api_version, azure_endpoint=azure_endpoint)

# =========================
# SYSTEM PROMPT v4.3 (prompt-only decisions; no dataset hardcoding)
# =========================
SYSTEM_PROMPT = r"""
SYSTEM PROMPT — Order Data Extractor (Compact v4.3, prompt-only)

ROLE
Extract structured order data from two plain-text inputs: RECEIPT (primary) and EMAIL (secondary). Return ONLY strict JSON per SCHEMA. If a value is missing or uncertain, return null. Do not guess.

SOURCE PRIORITY
- Buyer identity/contact (company, person, email): prefer EMAIL signature block if clearly the buyer; else RECEIPT; else null.
- Order header (order_number, order_date) + Delivery address: prefer RECEIPT ship-to block; fallback to EMAIL only if RECEIPT truly lacks those fields.
- Products: RECEIPT ONLY. Never take line items from EMAIL.

OUTPUT FORMAT
- Output a single JSON object exactly matching SCHEMA (no markdown, no comments, no extra keys).
- Preserve exact spelling/diacritics from the chosen source.

NORMALIZATION (for matching only; do NOT alter returned text)
- Trim spaces; collapse internal whitespace; case-fold for equality checks.
- Treat "&" ≈ "and"; "ß"≈"ss"; "ä/ae", "ö/oe", "ü/ue" as equivalent.
- For company matching, ignore trailing city tails like " - <City>" or ", <City>" during comparison.
- For order numbers, compare ignoring spaces (keep hyphens/letters).

VALIDATION
- Dates: parse many formats; output YYYY.MM.DD; if unparseable → null.
- Emails: must be a single valid address; else null. Avoid domain aliases (info@ or einkauf@) if a personal signature email exists nearby.
- Integers only for product_quantity and product_position.
- Postal codes: keep exactly as written (e.g., "40231", "W1A 1AA").

BUYER RULES
- buyer_company_name: take the legal name only (drop trailing location like " - Münster", ", Hamburg"). If EMAIL signature and RECEIPT differ trivially (e.g., "&" vs "and"), treat as same; prefer EMAIL signature form.
- buyer_person_name: the buyer contact (signature/contact line). If multiple, pick the clearest buyer contact; else null.
- buyer_email_address: prefer the email adjacent to the signature name; avoid group inboxes (info@, einkauf@) unless it’s the only address. If EMAIL and RECEIPT show different domains, choose the EMAIL signature domain.

ADDRESS RULES (SHIP-TO ONLY)
- Use the shipping/delivery block (labels: "Delivery", "Shipping", "Ship-to", "Lieferadresse", "Bitte liefern", "Versand an"). Never use vendor/company header or invoice addresses.
- If multiple addresses appear, choose the one explicitly labeled for delivery/ship-to OR the one immediately following such labels.
- If street/city/postal are on one line (e.g., "40231 Düsseldorf"), split into: postal="40231", city="Düsseldorf".
- Prefer the form with diacritics when variants exist (e.g., choose "Münster" over "Munster" if both appear near the ship-to block).
- If no reliable ship-to can be located, set street/city/postal to null rather than copying a header/footer address.

ORDER FIELDS
- order_number: digits or alphanumeric with hyphens; return exactly as written from RECEIPT if present; else EMAIL; else null.
- order_date: RECEIPT labels like "Order date", "Bestelldatum", "Datum" near the order header. Fallback to EMAIL only if clearly the creation date. Output YYYY.MM.DD or null.

PRODUCTS (RECEIPT ONLY)
- A line item is a row with a printed position number (labels/examples: "Pos.", "Pos", "Pos-", or a leading index column). Use that printed position as product_position.
- Extract each distinct purchased item row (don’t merge kit/container rows; don’t duplicate).

ARTICLE CODE SELECTION (choose exactly ONE token for each row):
Priority order when labels appear on/near the same row:
  1) Tokens under/near labels: "Ihre Materialnummer" | "Customer material no" | "Material-Nr Kunde".
  2) Else labels: Art.-Nr | Artikelnummer | Best.-Nr | Part No | Item | SKU.
  3) Else heuristic tie-breakers on the row: (a) a 5–10 digit numeric (e.g., 15630610), or (b) an alphanumeric starting with a letter (e.g., X920381742).
Deprioritize and only choose if explicitly labeled as the article number for the row:
  - Obvious family/catalog prefixes repeating across many rows (e.g., FRAI****, common vendor-family prefixes).
  - Unlabeled 12–14 digit numbers (likely GTIN/EAN) unless the label explicitly states Artikelnummer/Part No for that number.
If two candidates satisfy the same label, choose the one closest to the quantity cell on that row.

QUANTITY
- Use the integer from the row’s quantity cell/label ("Qty", "Menge", "Stk", "pcs"). Ignore pack size, price, or totals.

STRICT PRECEDENCE RECAP
- Buyer fields: EMAIL signature > RECEIPT.
- Order header + Delivery address: RECEIPT ship-to > EMAIL.
- Products: RECEIPT only.

FINAL CONSISTENCY PASS (YOU MUST DO THIS BEFORE OUTPUT)
- Check each selected field against the above rules. If a chosen address appears in header/footer boilerplate or lacks delivery labels, set address fields to null rather than guessing.
- If both diacritic and plain forms of a city exist, choose the diacritic form present near the ship-to block.
- For each line item, verify that product_position equals the printed Pos. number; if none found, exclude the row.
- For each line item, verify article_code follows the selection order and does not use family prefixes when a labeled numeric part exists.

SCHEMA (must match exactly)
{
  "buyer": {
    "buyer_company_name": "<string or null>",
    "buyer_person_name": "<string or null>",
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
# PDF text extraction (layout-aware when possible; still generic)
# =========================
def extract_text_from_pdf(pdf_path: str) -> str:
    # Try PyMuPDF (layout grouping keeps diacritics reliably)
    if fitz is not None:
        try:
            doc = fitz.open(pdf_path)
            parts = []
            for page in doc:
                # Blocks preserve reading order reasonably well
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
    # Last-resort: no extraction
    return ""

# =========================
# Prompt assembly
# =========================
def make_extraction_messages(email_text: str, receipt_text: str):
    # Light formatting to keep model guidance clear; no dataset-specific cues
    user_block = f"""USER INPUT (pass both blocks exactly):
EMAIL:
<<<{email_text or ""}>>>

RECEIPT:
<<<{receipt_text or ""}>>>"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_block},
    ]

# =========================
# LLM call (deterministic settings)
# =========================
def call_llm(messages: list) -> tuple[str, dict | None]:
    try:
        resp = client.responses.create(
            model=azure_deployment,
            input=messages,
            temperature=0,
            max_output_tokens=1500,
        )
        text = resp.output_text or ""
    except Exception as e:
        return f"LLM call failed: {e}", None

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
        if ok: match_count += 1
    acc = (match_count / total * 100.0) if total else 0.0
    return acc, rows

# =========================
# Build maps from CSVs (unchanged except diacritics fixes)
# =========================
def build_expected_map_from_csv(path: str) -> dict:
    df_exp = pd.read_csv(path, encoding="cp1252")
    expected_map: dict[int, dict] = {}

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
    emails = [str(v) for v in emails_series if isinstance(v, str) and v.strip() and not v.strip().lower().startswith('sales email')]
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

    for pdf_idx in tqdm(indices):
        pdf_path = os.path.join(PATH_RECEIPTS, f"{pdf_idx}.pdf")
        if not os.path.exists(pdf_path):
            print(f"PDF not found: {pdf_path}")
            continue
        email_text = email_map.get(pdf_idx, "")
        receipt_text = extract_text_from_pdf(pdf_path)
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
        results.append({
            "row": pdf_idx,
            "accuracy": acc,
            "ai_output": json_data,
            "expected": expected_json,
            "raw_ai": raw_text
        })
        print(f"\nRow {pdf_idx}:")
        print("AI Output:")
        print(json.dumps(json_data, ensure_ascii=False, indent=2))
        print("Expected Output:")
        print(json.dumps(expected_json, ensure_ascii=False, indent=2))
        print(f"Accuracy: {acc:.2f}%")

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
        acc_values.append(r["accuracy"])  # already percentage
    if acc_values:
        avg_acc = sum(acc_values) / len(acc_values)
        print(f"Average Accuracy: {avg_acc:.2f}%")
    print("\nWrote side-by-side comparison to batch_extraction_comparison.xlsx and summary CSV to batch_extraction_results.csv")

if __name__ == "__main__":
    main()
