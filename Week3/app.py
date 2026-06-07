import streamlit as st
import pandas as pd

st.title("Customer Intelligence System")

df = pd.read_csv("marketing_campaign.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Dataset Information")
st.write(f"Rows: {df.shape[0]}")
st.write(f"Columns: {df.shape[1]}")

st.subheader("Column Names")
st.write(df.columns.tolist())
