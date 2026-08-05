
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Interactive Data Dashboard")

df = pd.read_csv('1) iris.csv')

st.sidebar.header("Filter Options")
columns = df.columns.tolist()
selected_col = st.sidebar.selectbox("Select Column for Analysis", columns)

st.metric(label="Total Rows", value=len(df))

st.subheader("Interactive Distribution")
fig1 = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
st.plotly_chart(fig1)

st.subheader("Data Preview")
st.dataframe(df.head(10))