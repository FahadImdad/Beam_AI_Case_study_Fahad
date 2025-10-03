import os
import io
import json
import re
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# =========================
# Azure OpenAI client
# =========================
azure_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
if not all([azure_key, azure_api_version, azure_endpoint, azure_deployment]):
    st.error("Missing Azure OpenAI env vars. Please set AZURE_OPENAI_API_KEY (or AZURE_OPENAI_KEY), "
             "AZURE_OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT.")
client = AzureOpenAI(api_key=azure_key, api_version=azure_api_version, azure_endpoint=azure_endpoint)

st.set_page_config(page_title="EcomCorp Order Extractor", layout="wide")

# =========================
# SYSTEM PROMPT (insert your full version here)
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
# JSON Schema
# =========================
SCHEMA_JSON: dict = {
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
# Helpers
# =========================
def extract_text_from_pdf(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    text_parts = []
    try:
        b = uploaded_file.read() if hasattr(uploaded_file, 'read') else uploaded_file.getvalue()
        reader = PdfReader(io.BytesIO(b))
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
                if t:
                    text_parts.append(t)
            except Exception:
                pass
    except Exception as e:
        st.error(f"PDF read error: {e}")
    return "\n".join(text_parts).strip()

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

def call_llm(messages: list) -> tuple[str, dict | None]:
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
        except Exception:
            return f"Non-JSON or invalid JSON returned:\n{text}", None

# =========================
# UI
# =========================
st.title("EcomCorp Order Extractor")

left, right = st.columns([0.5, 0.5])

with left:
    uploaded_pdf = st.file_uploader("Upload Sales Order PDF", type=["pdf"])
    email_text = st.text_area("Paste Sales Email Body", height=220,
                              placeholder="From: ...\nSubject: ...\n\nDear ...")
    do_extract = st.button("Extract Order Data", type="primary")

receipt_text = ""
if uploaded_pdf:
    receipt_text = extract_text_from_pdf(uploaded_pdf)
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
            else:
                st.error("Could not parse JSON from the model. See raw output below.")
                st.code(raw_text, language="json")
