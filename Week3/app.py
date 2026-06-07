import streamlit as st
import pandas as pd
from pathlib import Path

st.title("Customer Intelligence System")

BASE_DIR = Path(__file__).parent
csv_path = BASE_DIR / "marketing_campaign.csv"

df = pd.read_csv(csv_path, sep="\t")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Dataset Information")
st.write(f"Rows: {df.shape[0]}")
st.write(f"Columns: {df.shape[1]}")

st.subheader("Column Names")
st.write(df.columns.tolist())
