import os
import io
import json
import pandas as pd
from tqdm import tqdm
from pypdf import PdfReader
from dotenv import load_dotenv
from google import genai
import re

# Load environment variables
load_dotenv()
client = genai.Client()

# Import or copy the SYSTEM_PROMPT and comparison logic from app.py
SYSTEM_PROMPT = """You are an elite information extractor. Output ONLY valid JSON matching the schema below—no extra text, no markdown, no comments. Use only values from the RECEIPT for all fields. Use exact spelling, punctuation, and special characters as in the RECEIPT (e.g., 'Münster', 'and' not '&'). If a value is missing or uncertain, use null. Never guess or hallucinate. Dates: output as YYYY.MM.DD if parseable, else null. Products must be a JSON array, not a dict. Do not include any human-readable summary.

SCHEMA (must match exactly):
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
"""

def extract_text_from_pdf(pdf_path) -> str:
    text_parts = []
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                try:
                    t = page.extract_text() or ""
                    text_parts.append(t)
                except Exception:
                    pass
    except Exception as e:
        print(f"PDF read error for {pdf_path}: {e}")
    return "\n".join([p for p in text_parts if p]).strip()

def make_extraction_messages(email_text: str, receipt_text: str):
    user_block = f"""USER INPUT (pass both blocks):
EMAIL:
<<<{email_text or ""}>>>

RECEIPT:
<<<{receipt_text or ""}>>>"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_block},
    ]

def call_llm(messages: list) -> tuple[str, dict | None]:
    try:
        prompt = "\n\n".join([m["content"] for m in messages])
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        text = response.text
    except Exception as e:
        return f"LLM call failed: {e}", None
    # Remove markdown code block if present
    if text.strip().startswith("```"):
        lines = text.strip().splitlines()
        if lines[0].startswith("```") and lines[-1].startswith("```"):
            text = "\n".join(lines[1:-1])
    try:
        data = json.loads(text)
        return text, data
    except json.JSONDecodeError as je:
        return f"Non-JSON or invalid JSON returned:\n{text}\n\nError: {je}", None

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

def parse_human_expected(text):
    # Remove leading/trailing quotes and whitespace
    text = text.strip().strip('"').replace('\r', '')
    # Replace double double-quotes with single double-quote (CSV escaping)
    text = text.replace('""', '"')
    buyer = {}
    order = {}
    products = []
    current = None
    product = {}
    for line in text.split('\n'):
        line = line.strip().replace('', 'ü').strip('"')
        if not line:
            continue
        lcline = line.lower().strip('"').strip()
        if lcline.startswith('buyer:'):
            current = 'buyer'
            continue
        if lcline.startswith('order:'):
            current = 'order'
            continue
        if lcline.startswith('product:'):
            if product and len(product) == 3:
                products.append(product)
            current = 'product'
            product = {}
            continue
        # For any line with a colon, assign to current section
        if ':' in line and current:
            key, value = line.split(':', 1)
            key = key.strip().lower().strip('"')
            value = value.strip().strip('"')
            if current == 'buyer':
                if key == 'buyer_company_name':
                    buyer['buyer_company_name'] = value or None
                elif key == 'buyer_person_name':
                    buyer['buyer_person_name'] = value or None
                elif key == 'buyer_email_address':
                    buyer['buyer_email_address'] = value or None
            elif current == 'order':
                if key == 'order_number':
                    order['order_number'] = value or None
                elif key == 'order_date':
                    import re
                    m = re.match(r'(\d{2})[.](\d{2})[.](\d{4})', value)
                    if m:
                        value = f"{m.group(3)}.{m.group(2)}.{m.group(1)}"
                    order['order_date'] = value or None
                elif key == 'delivery_address_street':
                    order['delivery_address_street'] = value or None
                elif key == 'delivery_address_city':
                    order['delivery_address_city'] = value or None
                elif key == 'delivery_address_postal_code':
                    order['delivery_address_postal_code'] = value or None
            elif current == 'product':
                if key == 'position':
                    try:
                        product['product_position'] = int(value)
                    except Exception:
                        product['product_position'] = None
                elif key == 'article_code':
                    product['product_article_code'] = value or None
                elif key == 'quantity':
                    try:
                        product['product_quantity'] = int(value)
                    except Exception:
                        product['product_quantity'] = None
                # If we have a complete product, append and reset
                if len(product) == 3:
                    products.append(product)
                    product = {}
    # At the end, append last product if it has data
    if current == 'product' and product and len(product) == 3:
        products.append(product)
    return {
        'buyer': buyer,
        'order': order,
        'products': products
    }

def main():
    df = pd.read_csv("Test Dataset - EcomCorp Order Extractor.csv", encoding="latin1")
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        pdf_idx = idx + 1  # Assuming 1-based PDF naming
        pdf_path = os.path.join("Sales Receipt", f"{pdf_idx}.pdf")
        if not os.path.exists(pdf_path):
            print(f"PDF not found: {pdf_path}")
            continue
        email_text = str(row["Sales Email"]) if not pd.isna(row["Sales Email"]) else ""
        expected_text = str(row["Expect Output"]) if not pd.isna(row["Expect Output"]) else ""
        if idx == 0:
            print("\nDEBUG: Raw expected_text for first row:")
            print(repr(expected_text))
        receipt_text = extract_text_from_pdf(pdf_path)
        messages = make_extraction_messages(email_text, receipt_text)
        raw_text, json_data = call_llm(messages)
        # Parse expected output (now using the parser)
        try:
            expected_json = parse_human_expected(expected_text)
        except Exception as e:
            print(f"Could not parse expected output for row {idx+1}: {e}")
            expected_json = None
        # Compare only the full dicts
        is_match = (json_data == expected_json)
        results.append({
            "row": idx+1,
            "match": is_match,
            "ai_output": json_data,
            "expected": expected_json,
            "raw_ai": raw_text
        })
        # Print side by side
        print(f"\nRow {idx+1}:")
        print("AI Output:")
        print(json.dumps(json_data, ensure_ascii=False, indent=2))
        print("Expected Output:")
        print(json.dumps(expected_json, ensure_ascii=False, indent=2))
        print(f"MATCH: {is_match}")
    # Save results as CSV
    out_rows = []
    for r in results:
        out_rows.append({
            "row": r["row"],
            "match": r["match"]
        })
    pd.DataFrame(out_rows).to_csv("batch_extraction_results.csv", index=False)

    # Save side-by-side comparison as a single Excel sheet
    side_by_side_rows = []
    for r in results:
        side_by_side_rows.append({
            "row": r["row"],
            "match": r["match"],
            "expected_output": json.dumps(r["expected"], ensure_ascii=False, indent=2) if r["expected"] is not None else None,
            "ai_output": json.dumps(r["ai_output"], ensure_ascii=False, indent=2) if r["ai_output"] is not None else None,
        })
    side_by_side_df = pd.DataFrame(side_by_side_rows, columns=["row", "match", "expected_output", "ai_output"])
    side_by_side_df.to_excel("batch_extraction_comparison.xlsx", index=False)

    # Print summary
    print("\nSummary:")
    for r in results:
        print(f"Row {r['row']}: MATCH={r['match']}")
    print("\nWrote side-by-side comparison to batch_extraction_comparison.xlsx and summary CSV to batch_extraction_results.csv")

if __name__ == "__main__":
    main()
