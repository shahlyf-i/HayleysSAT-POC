import streamlit as st
import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient

# ------------------------------
# Azure Custom Model Settings
# ------------------------------
endpoint = st.secrets["ENDPOINT"]
key = st.secrets["KEY"]
model_id = st.secrets["MODEL_ID"]

client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

# ------------------------------
# Helper function to deduplicate column names
# ------------------------------
def deduplicate_columns(columns):
    """
    Deduplicate column names by appending _1, _2, etc., to duplicates.
    Args:
        columns: Iterable of column names
    Returns:
        List of deduplicated column names
    """
    seen = {}
    result = []
    for col in map(str, columns):  # Convert all to strings
        if col in seen:
            seen[col] += 1
            result.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            result.append(col)
    return result

# ------------------------------
# Function to extract tables from PDF
# ------------------------------
def extract_tables_from_pdf(uploaded_file):
    """
    Extract tables from a PDF file using Azure Document Intelligence.
    Args:
        uploaded_file: Uploaded file object from Streamlit
    Returns:
        List of pandas DataFrames containing extracted tables
    """
    try:
        # Send file to Azure custom model
        poller = client.begin_analyze_document(model_id=model_id, body=uploaded_file)
        result = poller.result()
        tables_data = []
        if result.tables:
            for table in result.tables:
                max_row = max(cell.row_index for cell in table.cells) + 1
                max_col = max(cell.column_index for cell in table.cells) + 1
                grid = [["" for _ in range(max_col)] for _ in range(max_row)]
                for cell in table.cells:
                    grid[cell.row_index][cell.column_index] = cell.content
                df = pd.DataFrame(grid)
                if not df.empty:
                    # Use first row as columns and deduplicate
                    df.columns = deduplicate_columns(df.iloc[0].astype(str))
                    df = df[1:].reset_index(drop=True)
                tables_data.append(df)
        return tables_data
    except Exception as e:
        st.error(f"Error in extract_tables_from_pdf: {e}")
        return []

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Custom PDF Table Extractor (Azure Document Intelligence)")

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# ------------------------------
# Process uploaded PDFs
# ------------------------------
if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(f"Results for: {uploaded_file.name}")
        try:
            tables = extract_tables_from_pdf(uploaded_file)
            if tables:
                for table_idx, df in enumerate(tables):
                    st.write(f"Table {table_idx + 1} (Editable)")
                    
                    # Editable table
                    edited_df = st.data_editor(df, num_rows="dynamic")
                    
                    # Download button for edited table
                    csv = edited_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label=f"Download Edited Table {table_idx + 1} as CSV",
                        data=csv,
                        file_name=f"{uploaded_file.name}_table{table_idx + 1}_edited.csv",
                        mime="text/csv",
                    )
            else:
                st.warning("No tables detected in this PDF.")
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")