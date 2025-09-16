# app1.py
import streamlit as st
import pandas as pd
import math

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentAnalysisFeature

# -----------------------------
# Secrets
# -----------------------------
endpoint = st.secrets["ENDPOINT"]
key = st.secrets["KEY"]
model_id = st.secrets["MODEL_ID"]

client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

# -----------------------------
# Helpers
# -----------------------------
def deduplicate_columns(columns):
    seen, out = {}, []
    for c in map(str, columns):
        if c in seen:
            seen[c] += 1
            out.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out

def point_in_polygon(x, y, polygon):
    pts = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]; xj, yj = pts[j]
        if (yi > y) != (yj > y):
            x_int = (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
            if x < x_int:
                inside = not inside
        j = i
    return inside

def polygon_center(poly):
    xs, ys = poly[0::2], poly[1::2]
    return sum(xs)/len(xs), sum(ys)/len(ys)

def normalize_conf(v):
    """
    Normalize Azure confidence to 0..1 float.
    Accepts None, NaN, 0..1, or 0..100.
    """
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return math.nan
    try:
        v = float(v)
    except Exception:
        return math.nan
    # If already 0..1, keep it. If looks like 0..100, scale down.
    if v > 1.0:
        v = v / 100.0
    # Clamp
    v = max(0.0, min(1.0, v))
    return v

def conf_icon(pct, g=0.70, y=0.50):
    """
    Return colored icon string for confidence percentage (0..100).
    🟩 >= g*100, 🟨 >= y*100, else 🟥
    """
    if math.isnan(pct):
        return "—"
    if pct >= g * 100:
        return "🟩"
    if pct >= y * 100:
        return "🟨"
    return "🟥"

# -----------------------------
# Extract (values + confidence)
# -----------------------------
def extract_tables_with_confidence(uploaded_file):
    """
    Returns: list of {"values": DataFrame[str], "scores": DataFrame[float 0..1 or NaN]}
    Uses cell.confidence when available; otherwise falls back to OCR words.
    """
    pdf_bytes = uploaded_file.read()
    poller = client.begin_analyze_document(
        model_id=model_id,
        body=pdf_bytes,
        content_type="application/pdf",
        features=[DocumentAnalysisFeature.OCR_HIGH_RESOLUTION]
    )
    result = poller.result()

    if not result.tables:
        return []

    pages = {p.page_number: p for p in (result.pages or [])}
    out = []

    for table in result.tables:
        max_row = max(c.row_index for c in table.cells) + 1
        max_col = max(c.column_index for c in table.cells) + 1

        grid_vals = [["" for _ in range(max_col)] for _ in range(max_row)]
        grid_conf = [[math.nan for _ in range(max_col)] for _ in range(max_row)]

        for cell in table.cells:
            r, c = cell.row_index, cell.column_index
            grid_vals[r][c] = cell.content or ""

            # 1) Try model-provided confidence
            conf = getattr(cell, "confidence", None)

            # 2) Fallback to OCR words inside the cell polygon (mean)
            if (conf is None) and cell.bounding_regions:
                br = cell.bounding_regions[0]
                page = pages.get(br.page_number)
                poly = br.polygon
                if page and page.words and poly:
                    scores = []
                    for w in page.words:
                        if w.polygon and (w.confidence is not None):
                            cx, cy = polygon_center(w.polygon)
                            if point_in_polygon(cx, cy, poly):
                                scores.append(float(w.confidence))
                    if scores:
                        conf = sum(scores) / len(scores)  # mean is smoother than min

            grid_conf[r][c] = normalize_conf(conf)

        df_vals = pd.DataFrame(grid_vals)
        df_conf = pd.DataFrame(grid_conf)

        # Promote first row to header
        if not df_vals.empty:
            df_vals.columns = deduplicate_columns(df_vals.iloc[0].astype(str))
            df_conf.columns = df_vals.columns
            df_vals = df_vals[1:].reset_index(drop=True)
            df_conf = df_conf[1:].reset_index(drop=True)

        out.append({"values": df_vals, "scores": df_conf})

    return out

# -----------------------------
# Build editor with icons
# -----------------------------
def build_editor_with_conf(df_vals: pd.DataFrame, df_conf: pd.DataFrame, green_cut=0.70, yellow_cut=0.50):
    """
    Interleave columns: Value, Value (conf) where the conf column is a read-only text like '78% 🟩'.
    """
    data = {}
    order = []
    for col in df_vals.columns:
        # values
        data[col] = df_vals[col]
        order.append(col)

        # confidence (text + icon)
        conf_col = f"{col} (conf)"
        vals = []
        for v in df_conf[col].tolist():
            pct = (v * 100.0) if pd.notna(v) else float('nan')
            icon = conf_icon(pct, g=green_cut, y=yellow_cut)
            if pd.isna(pct):
                vals.append("—")
            else:
                vals.append(f"{int(round(pct))}% {icon}")
        data[conf_col] = vals
        order.append(conf_col)

    editor_df = pd.DataFrame(data)[order]
    # Mark only the conf columns as disabled
    disabled_cols = [c for c in editor_df.columns if c.endswith("(conf)")]
    # Column config: make conf text non-editable and fixed width
    col_config = {c: st.column_config.TextColumn(c, disabled=True) for c in disabled_cols}
    return editor_df, disabled_cols, col_config

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="PDF Table Extractor (with Confidence)", layout="wide")
st.title("Custom PDF Table Extractor (Azure Document Intelligence)")

uploaded_files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    # Thresholds (you asked green at ≥70%)
    green_cut = st.slider("Green at ≥ this %", 50, 95, 70, 1) / 100.0
    yellow_cut = st.slider("Yellow lower bound %", 30, 69, 50, 1) / 100.0

    for uploaded_file in uploaded_files:
        st.subheader(f"Results for: {uploaded_file.name}")

        tables = extract_tables_with_confidence(uploaded_file)
        if not tables:
            st.warning("No tables detected in this PDF.")
            continue

        for i, t in enumerate(tables, start=1):
            df_vals, df_conf = t["values"], t["scores"]

            pct_scored = (df_conf.notna().sum().sum() / df_conf.size * 100) if df_conf.size else 0
            st.caption(f"Cells with confidence available: {pct_scored:.1f}%")

            st.markdown(f"**Table {i} – Editable (values + confidence)**")
            editor_df, disabled_cols, conf_config = build_editor_with_conf(
                df_vals, df_conf, green_cut=green_cut, yellow_cut=yellow_cut
            )

            edited_df = st.data_editor(
                editor_df,
                num_rows="dynamic",
                use_container_width=True,
                disabled=disabled_cols,
                column_config=conf_config
            )

            # Download: values only (strip the confidence columns)
            value_cols = [c for c in edited_df.columns if not c.endswith("(conf)")]
            clean_values = edited_df[value_cols]
            st.download_button(
                label=f"Download Edited Table {i} (values only) as CSV",
                data=clean_values.to_csv(index=False).encode("utf-8"),
                file_name=f"{uploaded_file.name}_table{i}_edited.csv",
                mime="text/csv",
            )