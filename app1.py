# app1.py — In-cell highlights (yellow/red only) with readable text
# Finetuned sanitizers:
#  - MC  : keep LEFT-MOST number only (e.g., "22 84" -> "22")
#  - RPM : keep FIRST TWO DIGITS of the LEFT-MOST number (e.g., "2050" -> "20")

import re
import math
import pandas as pd
import streamlit as st

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentAnalysisFeature

from st_aggrid import (
    AgGrid,
    GridOptionsBuilder,
    GridUpdateMode,
    JsCode,
    AgGridTheme,
)

# -----------------------------
# Secrets (set in .streamlit/secrets.toml)
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
    pts = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
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
    return sum(xs) / len(xs), sum(ys) / len(ys)

def normalize_conf(v):
    """Normalize Azure confidence to 0..1; NaN if missing."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return math.nan
    try:
        v = float(v)
    except Exception:
        return math.nan
    if v > 1.0:
        v /= 100.0
    return max(0.0, min(1.0, v))

# ---------- NEW: sanitizers ----------
def _leftmost_digits(text: str) -> str:
    """
    Return the left-most run of digits in text (e.g., '22 84' -> '22').
    If no digits are found, return the original text unchanged.
    """
    s = str(text)
    m = re.search(r'\d+', s)
    return m.group(0) if m else s

def _leftmost_two_digits(text: str) -> str:
    """
    Take the left-most run of digits, then keep only its first two digits.
    Examples: '2050' -> '20', '22 24' -> '22', '8A' -> '8'
    If no digits are found, return the original text unchanged.
    """
    first = _leftmost_digits(text)
    return first[:2] if first and first[0].isdigit() else first

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply per-column cleanup:
      - MC  : left-most number only
      - RPM : first two digits of the left-most number
    Matches columns case-insensitively and allows suffixes like '_1'.
    """
    if df.empty:
        return df

    for col in df.columns:
        name = str(col).strip().lower()
        if re.fullmatch(r'mc(?:_\d+)?', name):
            df[col] = df[col].astype(str).map(_leftmost_digits)
        elif re.fullmatch(r'rpm(?:_\d+)?', name):
            df[col] = df[col].astype(str).map(_leftmost_two_digits)
    return df
# ------------------------------------

# -----------------------------
# Extract (values + confidence)
# -----------------------------
def extract_tables_with_confidence(uploaded_file):
    """
    Returns list of {"values": df_vals (str), "scores": df_conf (0..1 or NaN)}.
    Uses cell.confidence when available; otherwise estimates from OCR word confidences.
    """
    pdf_bytes = uploaded_file.read()
    poller = client.begin_analyze_document(
        model_id=model_id,
        body=pdf_bytes,
        content_type="application/pdf",
        features=[DocumentAnalysisFeature.OCR_HIGH_RESOLUTION],
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

            # 1) model-provided confidence
            conf = getattr(cell, "confidence", None)

            # 2) fallback: average word confidences inside the cell polygon
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
                        conf = sum(scores) / len(scores)

            grid_conf[r][c] = normalize_conf(conf)

        df_vals = pd.DataFrame(grid_vals)
        df_conf = pd.DataFrame(grid_conf)

        # Promote first row to header
        if not df_vals.empty:
            df_vals.columns = deduplicate_columns(df_vals.iloc[0].astype(str))
            df_conf.columns = df_vals.columns
            df_vals = df_vals[1:].reset_index(drop=True)
            df_conf = df_conf[1:].reset_index(drop=True)

            # ---------- apply the new sanitizers here ----------
            df_vals = sanitize_columns(df_vals)

        out.append({"values": df_vals, "scores": df_conf})

    return out

# -----------------------------
# AgGrid editor with in-cell highlight (readable text)
# -----------------------------
def aggrid_editor_colored(df_vals: pd.DataFrame, df_conf: pd.DataFrame,
                          yellow_cut=0.50, green_cut=0.70, height=460):
    """
    Editable value cells with background highlight:
      - < yellow_cut  → red  (#FFCDD2) with black bold text
      - yellow_cut..green_cut  → yellow (#FFF9C4) with black bold text
      - ≥ green_cut → no color (no green)
    Confidence is stored in hidden columns __conf__<col>.
    """
    merged = df_vals.copy()
    for col in df_vals.columns:
        merged[f"__conf__{col}"] = df_conf[col]

    js_style = JsCode(f"""
        function(params){{
            var key = "__conf__" + params.colDef.field;
            var v = params.data[key];
            if (v === null || v === undefined || isNaN(v)) return {{}};
            if (v >= {green_cut}) return {{}}; // high → no highlight
            if (v >= {yellow_cut}) {{
                return {{'backgroundColor': '#FFF9C4', 'color': '#111111', 'fontWeight': '600',
                         'border': '1px solid #9CA3AF'}}; // yellow, black bold text
            }}
            return {{'backgroundColor': '#FFCDD2', 'color': '#111111', 'fontWeight': '600',
                     'border': '1px solid #9CA3AF'}};     // red, black bold text
        }}
    """)

    gob = GridOptionsBuilder.from_dataframe(merged, editable=True)
    for col in df_vals.columns:
        gob.configure_column(col, editable=True, cellStyle=js_style)
        gob.configure_column(f"__conf__{col}", hide=True, editable=False)

    grid = AgGrid(
        merged,
        gridOptions=gob.build(),
        update_mode=GridUpdateMode.VALUE_CHANGED,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True,
        theme=AgGridTheme.STREAMLIT,   # valid: STREAMLIT, BALHAM, ALPINE, MATERIAL
        height=height,
    )
    edited = pd.DataFrame(grid["data"])
    value_cols = [c for c in edited.columns if not c.startswith("__conf__")]
    return edited[value_cols]

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="PDF Table Extractor (in-cell highlights)", layout="wide")
st.title("Custom PDF Table Extractor (Azure Document Intelligence)")

files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

if files:
    # Only yellow/red; green is intentionally not shown
    green_cut = st.slider("High-confidence threshold (no color when ≥)", 60, 95, 70, 1) / 100.0
    yellow_cut = st.slider("Yellow threshold (≥)", 30, 69, 50, 1) / 100.0

    for up in files:
        st.subheader(f"Results for: {up.name}")
        tables = extract_tables_with_confidence(up)
        if not tables:
            st.warning("No tables detected in this PDF.")
            continue

        for i, t in enumerate(tables, start=1):
            df_vals, df_conf = t["values"], t["scores"]
            pct_scored = (df_conf.notna().sum().sum() / df_conf.size * 100) if df_conf.size else 0
            st.caption(f"Cells with confidence available: {pct_scored:.1f}%")

            st.markdown(f"**Table {i} – Editable (in-cell highlights: yellow/red only)**")
            edited_values = aggrid_editor_colored(
                df_vals, df_conf, yellow_cut=yellow_cut, green_cut=green_cut
            )

            st.download_button(
                f"Download Edited Table {i} as CSV",
                edited_values.to_csv(index=False).encode("utf-8"),
                file_name=f"{up.name}_table{i}_edited.csv",
                mime="text/csv",
            )