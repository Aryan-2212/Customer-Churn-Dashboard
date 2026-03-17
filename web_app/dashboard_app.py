import streamlit as st

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("AI-Powered Customer Churn Dashboard")

st.write("""
This dashboard provides insights into customer churn behavior.

Use the AI assistant below to ask questions about the dashboard insights.
""")

st.subheader("Power BI Dashboard")

st.markdown(
"""
<iframe title="Churn Dashboard"
width="100%"
height="600"
src="PASTE_POWERBI_EMBED_LINK_HERE"
frameborder="0"
allowFullScreen="true">
</iframe>
""",
unsafe_allow_html=True
)