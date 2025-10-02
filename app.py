import os
import io
import json
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
import pandas as pd
from openai import AzureOpenAI

load_dotenv()

# Azure OpenAI client
azure_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
if not all([azure_key, azure_api_version, azure_endpoint, azure_deployment]):
    st.error("Missing Azure OpenAI env vars. Please set AZURE_OPENAI_API_KEY (or AZURE_OPENAI_KEY), AZURE_OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT.")
client = AzureOpenAI(api_key=azure_key, api_version=azure_api_version, azure_endpoint=azure_endpoint)

st.set_page_config(page_title="EcomCorp Order Extractor", layout="wide")

SYSTEM_PROMPT = """You extract structured order data. Prefer RECEIPT over EMAIL for overlaps. If a value is missing or uncertain, return null. Do not guess.

Output ONLY strict JSON matching the schema (no extra keys/text/markdown). Keep exact spelling/diacritics from sources. Dates: accept many formats; output YYYY.MM.DD if parseable, else null.

Heuristics (general, not absolute):
- product_article_code: pick the primary product code token on the line. If multiple candidates appear, prefer an alphanumeric code starting with a letter (e.g., X7630260, P8420610). If only a numeric code exists (e.g., 15630610), use it as written. Strip obvious trailing descriptors (units/notes like EA/QTY/Dxx/E) when clearly separable from the code.
- buyer_email_address: must be a valid email if present; else null.
- order_number: may be digits or mixed with hyphens (e.g., 2024-08477). Return as written from RECEIPT.
- delivery_address_city/postal: preserve diacritics; postal codes are often 5 digits; if ambiguous, return as written or null.
- products: include each explicit line item from RECEIPT only; 1-based positions in reading order; no inferred or duplicate items.

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
    # Using Azure OpenAI Responses API
    try:
        resp = client.responses.create(
            model=azure_deployment,
            input=messages,
            temperature=0,
            max_output_tokens=1200,
        )
        text = resp.output_text or ""
    except Exception as e:
        return f"LLM call failed: {e}", None

    # Remove markdown code block if present
    if text.strip().startswith("```"):
        lines = text.strip().splitlines()
        if lines[0].startswith("```") and lines[-1].startswith("```"):
            text = "\n".join(lines[1:-1])

    # Fallback: try to extract JSON substring
    import re as _re
    candidate = text.strip()
    if not candidate.startswith("{"):
        m = _re.search(r"\{[\s\S]*\}\s*$", candidate)
        if m:
            candidate = m.group(0)
    try:
        data = json.loads(candidate)
        return candidate, data
    except json.JSONDecodeError:
        m2 = _re.search(r"\{[\s\S]*\}", text)
        if m2:
            try:
                data = json.loads(m2.group(0))
                return m2.group(0), data
            except Exception:
                pass
        return f"Non-JSON or invalid JSON returned:\n{text}", None

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
