# app.py
import streamlit as st
import pdfplumber
import pandas as pd
from io import BytesIO
import re
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
import subprocess
import tempfile
import os

st.set_page_config(page_title="Financial Q&A (Final)", layout="wide")
st.title("📊 Financial Document Q&A System")

# -------------------------
# Helpers
# -------------------------
def normalize_num(s):
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s.lower() in ["nan", "none"]:
        return None
    if re.match(r'^\(.*\)$', s):
        s = '-' + s.strip('()')
    s = re.sub(r'[^\d\.\-]', '', s)
    try:
        return float(s)
    except:
        return None

def extract_from_pdf_bytes(b: bytes) -> Dict:
    text = ""
    tables = []
    with pdfplumber.open(BytesIO(b)) as pdf:
        for page in pdf.pages:
            ptext = page.extract_text() or ""
            text += ptext + "\n"
            for t in page.extract_tables():
                try:
                    df = pd.DataFrame(t[1:], columns=t[0])
                    tables.append(df)
                except Exception:
                    tables.append(pd.DataFrame(t))
    return {"text": text, "tables": tables}

def extract_from_excel_bytes(b: bytes) -> Dict:
    xls = pd.read_excel(BytesIO(b), sheet_name=None)
    text = ""
    tables = []
    for name, df in xls.items():
        text += f"Sheet: {name}\n"
        tables.append(df)
    return {"text": text, "tables": tables}

def tables_to_rows(tables: List[pd.DataFrame]) -> List[Dict]:
    rows = []
    for df in tables:
        df.columns = [str(c) for c in df.columns]
        for _, row in df.iterrows():
            account = None
            value = None
            for col in df.columns:
                cell = row[col]
                if isinstance(cell, str) and cell.strip():
                    if account is None:
                        account = cell.strip()
                        continue
                nv = normalize_num(cell)
                if nv is not None and value is None:
                    value = nv
            if account:
                rows.append({"account": account, "value": value})
    return rows

# -------------------------
# Embedding model
# -------------------------
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
embed_model = load_embed_model()

def embed_texts(texts: List[str]):
    return embed_model.encode(texts, convert_to_numpy=True)

def cosine_search(query: str, corpus_texts: List[str], corpus_embs: np.ndarray, top_k: int=5):
    q_emb = embed_model.encode([query], convert_to_numpy=True)[0]
    denom = (np.linalg.norm(corpus_embs, axis=1) * np.linalg.norm(q_emb) + 1e-10)
    scores = (corpus_embs @ q_emb) / denom
    idx = np.argsort(-scores)[:top_k]
    return [(int(i), float(scores[i]), corpus_texts[int(i)]) for i in idx]

# -------------------------
# Ollama CLI call with fallback (prompt-file)
# -------------------------
def call_ollama(model: str, prompt: str, timeout: int=180) -> str:
    """Call Ollama by piping prompt via stdin (compatible)."""
    try:
        proc = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout
        )
        if proc.returncode == 0:
            return proc.stdout.strip() or proc.stderr.strip()
        else:
            return f"Ollama error {proc.returncode}: {proc.stderr.strip()}"
    except Exception as e:
        return f"Ollama call failed: {e}"
# -------------------------
# Streamlit App
# -------------------------
uploaded = st.file_uploader("Upload PDF or Excel file", type=["pdf", "xlsx", "xls"])
if uploaded:
    st.success(f"File `{uploaded.name}` uploaded successfully!")
    b = uploaded.read()
    if uploaded.name.lower().endswith(".pdf"):
        doc = extract_from_pdf_bytes(b)
    else:
        doc = extract_from_excel_bytes(b)

    st.subheader("📄 Extracted text (first 2000 chars)")
    st.code(doc["text"][:2000])

    st.subheader("📊 Extracted tables (first 3)")
    for i, df in enumerate(doc["tables"][:3]):
        st.write(f"Table {i+1}")
        st.dataframe(df.fillna(""))

    rows = tables_to_rows(doc["tables"])
    st.write("Detected rows:", len(rows))

    if len(rows) > 0:
        norm_df = pd.DataFrame(rows)
        st.dataframe(norm_df)

        # Chunks for embeddings
        chunks = [doc["text"][:1500]] if doc["text"].strip() else []
        for r in rows:
            chunks.append(f"{r['account']} => {r['value']}")

        with st.spinner("Computing embeddings..."):
            corpus_embs = embed_texts(chunks)
        st.success("Embeddings ready ✅")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        st.subheader("💬 Ask a Question")
        user_q = st.text_input("Type your question here")
        model_to_use = st.text_input("Ollama model name", value="mistral")

        if st.button("Ask"):
            if not user_q.strip():
                st.warning("Please type a question.")
            else:
                # 1) Structured direct lookup
                q_lower = user_q.lower()
                numeric_hit = None
                for r in rows:
                    if any(k in r["account"].lower() for k in q_lower.split()):
                        numeric_hit = r
                        break

                if numeric_hit and numeric_hit.get("value") is not None:
                    answer = f"Found `{numeric_hit['account']}` → **{int(numeric_hit['value']):,}**"
                else:
                    # 2) Retrieval + Ollama reasoning
                    top = cosine_search(user_q, chunks, corpus_embs, top_k=5)
                    context = "\n\n".join([f"{t}" for (_, _, t) in top])
                    system_instructions = (
                        "You are a financial assistant.\n"
                        "Use ONLY the given context to answer the user's question.\n"
                        "If a calculation is possible (e.g., revenue - expenses), perform it.\n"
                        "Always mention which values you used from the context.\n"
                        "If you cannot find it, say 'Not found in document'."
                    )
                    prompt = f"{system_instructions}\n\nContext:\n{context}\n\nQuestion: {user_q}\n\nAnswer:"
                    answer = call_ollama(model_to_use, prompt)

                st.session_state.messages.append({"q": user_q, "a": answer})

        if st.session_state.messages:
            st.markdown("### Conversation")
            for m in reversed(st.session_state.messages[-10:]):
                st.markdown(f"**Q:** {m['q']}")
                st.markdown(f"**A:** {m['a']}")
