import os
import io
import json
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
import pandas as pd
from google import genai

load_dotenv()
# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

st.set_page_config(page_title="EcomCorp Order Extractor", layout="wide")

SYSTEM_PROMPT = """You are an elite information extractor. Use only values from the RECEIPT, not EMAIL, for all fields. Output valid JSON matching the schema exactly—never add or remove keys. Use exact spelling, punctuation, and special characters as in the RECEIPT (e.g., 'Münster' not 'Munster', 'and' not '&'). If a value is missing or uncertain, use null. Never guess or hallucinate. Dates: output as YYYY.MM.DD if parseable, else null. Strict JSON only—no comments, no markdown, no trailing commas.

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

def extract_text_from_pdf(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    text_parts = []
    try:
        reader = PdfReader(io.BytesIO(uploaded_file.read() if hasattr(uploaded_file, 'read') else uploaded_file.getvalue()))
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
                text_parts.append(t)
            except Exception:
                pass
    except Exception as e:
        st.error(f"PDF read error: {e}")
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
        # Remove the first and last line if they are code block markers
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
        rows.append((k, e, g, "✅" if ok else "❌"))
        total += 1
        if ok: match_count += 1
    acc = (match_count / total * 100.0) if total else 0.0
    return acc, rows

st.title("EcomCorp Order Extractor")

left, right = st.columns([0.5, 0.5])

with left:
    uploaded_pdf = st.file_uploader("Upload Sales Order PDF", type=["pdf"])
    email_text = st.text_area("Paste Sales Email Body", height=220, placeholder="From: ...\nSubject: ...\n\nDear ...")
    expected_text = st.text_area("Paste Expected JSON (optional, for accuracy comparison)", height=220, placeholder='{"buyer": {...}, "order": {...}, "products": [...]}')
    do_extract = st.button("Extract Order Data", type="primary")

receipt_text = ""
if uploaded_pdf:
    pdf_bytes = uploaded_pdf.getvalue()
    receipt_text = extract_text_from_pdf(io.BytesIO(pdf_bytes))
    if not receipt_text:
        st.warning("No text extracted. If this is a scanned PDF, enable OCR in your pipeline (not included in this demo).")

if do_extract:
    if not uploaded_pdf and not email_text.strip():
        st.error("Please upload a PDF and/or paste the email text.")
    else:
        messages = make_extraction_messages(email_text.strip(), receipt_text)
        raw_text, json_data = call_llm(messages)
        with right:
            st.subheader("Receipt Text (extracted)")
            st.text_area("Receipt text", receipt_text or "(none)", height=220)
            st.subheader("Extraction Result (JSON)")
            if json_data is not None:
                st.json(json_data)
                st.download_button(
                    "Download JSON",
                    data=json.dumps(json_data, ensure_ascii=False, indent=2),
                    file_name="extracted_order.json",
                    mime="application/json",
                )
                if expected_text.strip():
                    try:
                        expected_json = json.loads(expected_text)
                        acc, rows = compare_json(expected_json, json_data)
                        st.markdown(f"**Variable-level accuracy:** {acc:.2f}%")
                        df = pd.DataFrame(rows, columns=["path", "expected", "got", "match"])
                        st.dataframe(df, use_container_width=True, height=320)
                    except Exception as e:
                        st.error(f"Could not parse Expected JSON: {e}")
            else:
                st.error("Could not parse JSON from the model. See raw output below.")
                st.code(raw_text, language="json")
