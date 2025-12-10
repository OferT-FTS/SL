import streamlit as st

st.set_page_config(
    page_title="Final Data Science Project",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

# Header with emoji
st.markdown("""
    <h1 style='text-align: center; color: gold; font-weight: 350'>
        Research Problem Description
    </h1>
""", unsafe_allow_html=True)

st.sidebar.markdown("## Research Problem Description")

# Spacing
st.markdown("<br>", unsafe_allow_html=True)