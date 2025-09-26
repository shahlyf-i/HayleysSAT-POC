# app1.py — Editable tables with deferred "Save Changes" + cached extraction + download after save

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
    DataReturnMode,
)

# -----------------------------
# Secrets
# -----------------------------
st.set_page_config(page_title="PDF Table Extractor (yellow/red highlights)", layout="wide")
endpoint = st.secrets["ENDPOINT"]
key = st.secrets["KEY"]
model_id = st.secrets["MODEL_ID"]
client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

# -----------------------------
# Session state init  ✅
# -----------------------------
if "edited_tables" not in st.session_state:
    # structure: st.session_state.edited_tables[(filename, table_index)] = pd.DataFrame
    st.session_state.edited_tables = {}
if "extract_cache_debug" not in st.session_state:
    st.session_state.extract_cache_debug = {}

# -----------------------------
# Helpers (unchanged)
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
    if v is None:
        return math.nan
    try:
        v = float(v)
    except Exception:
        return math.nan
    if v > 1.0:
        v /= 100.0
    return max(0.0, min(1.0, v))

def _leftmost_digits(text: str) -> str:
    m = re.search(r'\d+', str(text))
    return m.group(0) if m else str(text)

def _leftmost_two_digits(text: str) -> str:
    first = _leftmost_digits(text)
    return first[:2] if first and first[0].isdigit() else first

def sanitize_mc_rpm(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for col in df.columns:
        name = str(col).strip().lower()
        if name == "mc" or re.fullmatch(r"mc_\d+", name):
            df[col] = df[col].astype(str).map(_leftmost_digits)
        elif name == "rpm" or re.fullmatch(r"rpm_\d+", name):
            df[col] = df[col].astype(str).map(_leftmost_two_digits)
    return df

def _topmost_numeric_from_words(page, polygon, min_len=3, max_len=5):
    if not page or not getattr(page, "words", None) or not polygon:
        return ""
    candidates = []
    for w in page.words:
        if not (w.polygon and w.content):
            continue
        s = w.content.strip()
        if not re.fullmatch(rf"\d{{{min_len},{max_len}}}", s):
            continue
        cx, cy = polygon_center(w.polygon)
        if point_in_polygon(cx, cy, polygon):
            candidates.append((cy, cx, s))
    if not candidates:
        return ""
    candidates.sort(key=lambda t: (t[0], t[1]))
    return candidates[0][2]

# -----------------------------
# Cached extraction  ✅
# -----------------------------
@st.cache_data(show_spinner=False, max_entries=64)
def extract_tables_with_confidence_from_bytes(pdf_bytes: bytes):
    """
    Cached heavy call. Keyed by bytes content (safe for small/medium PDFs).
    Returns list of {values: df, scores: df}
    """
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
            value = (cell.content or "").strip()

            if len(re.findall(r"\d{3,5}", value)) >= 2 and cell.bounding_regions:
                br = cell.bounding_regions[0]
                page = pages.get(br.page_number)
                chosen = _topmost_numeric_from_words(page, br.polygon, 3, 5)
                if chosen:
                    value = chosen

            grid_vals[r][c] = value

            conf = getattr(cell, "confidence", None)
            if conf is None and cell.bounding_regions:
                br = cell.bounding_regions[0]
                page = pages.get(br.page_number)
                poly = br.polygon
                if page and getattr(page, "words", None) and poly:
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

        if not df_vals.empty:
            df_vals.columns = deduplicate_columns(df_vals.iloc[0].astype(str))
            df_conf.columns = df_vals.columns
            df_vals = df_vals[1:].reset_index(drop=True)
            df_conf = df_conf[1:].reset_index(drop=True)
            df_vals = sanitize_mc_rpm(df_vals)

        out.append({"values": df_vals, "scores": df_conf})

    return out

# -----------------------------
# AG-Grid editor (unchanged core); we don't persist here  ✅
# -----------------------------
def aggrid_editor_yellow_red(df_vals, df_conf, yellow_cut=0.50, green_cut=0.70, height=460, grid_key=None):
    if df_vals.empty:
        return df_vals.copy()

    orig_cols = list(df_vals.columns)
    safe_cols, used = [], set()
    for c in orig_cols:
        k = re.sub(r"[^A-Za-z0-9_]", "_", str(c)) or "col"
        while k in used:
            k += "_"
        used.add(k)
        safe_cols.append(k)

    vals_display = df_vals.copy()
    vals_display.columns = safe_cols

    conf_display = df_conf.copy()
    conf_display.columns = [f"__conf__{k}" for k in safe_cols]

    merged = pd.concat([vals_display, conf_display], axis=1)

    js_style = JsCode(f"""
        function(params){{
            var k = "__conf__" + params.colDef.field;
            var v = params.data[k];
            if (v === null || v === undefined || isNaN(v)) return {{}};
            if (v >= {green_cut}) return {{}};
            if (v >= {yellow_cut}) {{
                return {{backgroundColor: '#FFF9C4', color: '#111', fontWeight: '600',
                         border: '1px solid #9CA3AF'}};
            }}
            return {{backgroundColor: '#FFCDD2', color: '#111', fontWeight: '600',
                     border: '1px solid #9CA3AF'}};
        }}
    """)

    gob = GridOptionsBuilder.from_dataframe(merged, editable=True)
    for safe, orig in zip(safe_cols, orig_cols):
        gob.configure_column(safe, header_name=str(orig), editable=True, cellStyle=js_style)
        gob.configure_column(f"__conf__{safe}", hide=True, editable=False)

    gob.configure_grid_options(stopEditingWhenCellsLoseFocus=True)

    grid = AgGrid(
        merged,
        gridOptions=gob.build(),
        data_return_mode=DataReturnMode.AS_INPUT,
        # Important: edits may cause reruns, but caching prevents heavy work
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True,
        theme=AgGridTheme.STREAMLIT,
        height=height,
        key=grid_key,
    )

    edited = pd.DataFrame(grid["data"])
    edited_vals = edited[safe_cols].copy()
    edited_vals.columns = orig_cols
    return edited_vals

# -----------------------------
# UI
# -----------------------------
st.title("Custom PDF Table Extractor (Azure Document Intelligence)")

files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

if files:
    green_cut = st.slider("No color when ≥", 60, 95, 70, 1) / 100.0
    yellow_cut = st.slider("Yellow when ≥", 30, 69, 50, 1) / 100.0

    for up in files:
        pdf_bytes = up.getvalue()  # ✅ stable bytes for cache key
        # cache hit indicator (optional)
        tables = extract_tables_with_confidence_from_bytes(pdf_bytes)
        st.subheader(f"Results for: {up.name}")
        if not tables:
            st.warning("No tables detected in this PDF.")
            continue

        for i, t in enumerate(tables, start=1):
            df_vals, df_conf = t["values"], t["scores"]
            pct_scored = (df_conf.notna().sum().sum() / df_conf.size * 100) if df_conf.size else 0
            st.caption(f"Cells with confidence available: {pct_scored:.1f}%")

            st.markdown(f"**Table {i} – Edit cells, then click _Save Changes_**")
            grid_key = f"{up.name}_table_{i}"

            # Use an expander + form so the *save* is an explicit action  ✅
            with st.expander(f"Open Table {i}", expanded=True):
                with st.form(key=f"form_{grid_key}"):
                    edited_values_live = aggrid_editor_yellow_red(
                        df_vals, df_conf,
                        yellow_cut=yellow_cut,
                        green_cut=green_cut,
                        grid_key=grid_key,
                    )

                    # Only persist when the user clicks this  ✅
                    saved = st.form_submit_button("Save Changes")

                    if saved:
                        st.session_state.edited_tables[(up.name, i)] = edited_values_live
                        st.success("Changes saved for this table.")

            # If saved before, show the saved preview and download  ✅
            saved_key = (up.name, i)
            if saved_key in st.session_state.edited_tables:
                saved_df = st.session_state.edited_tables[saved_key]
                st.dataframe(saved_df, use_container_width=True)
                st.download_button(
                    f"Download Edited Table {i} as CSV",
                    saved_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{up.name}_table{i}_edited.csv",
                    mime="text/csv",
                    key=f"dl_{grid_key}",
                )
            else:
                # Not saved yet — let user know  ✅
                st.info("Edit cells above, then click **Save Changes** to enable download.")