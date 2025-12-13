import streamlit as st

st.set_page_config(
    page_title="References",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

# Header with emoji
st.markdown("""
    <h1 style='text-align: center; color: gold; font-weight: 350'>
        References
    </h1>
""", unsafe_allow_html=True)

st.sidebar.markdown("## References")

# Spacing
st.markdown("<br>", unsafe_allow_html=True)

# --- Use a container for the main content to keep it organized ---
with st.container():
    st.markdown(
        "Here are the sources used throughout this presentation. For more information, please refer to the following materials:")  #

    st.markdown("---")  # Add a horizontal line for visual separation

    # --- Reference List Section ---
    # Use st.markdown with bullet points for a clean, readable list format.
    # You can format the citations using standard academic styles (e.g., APA, MLA, Chicago).

    st.subheader("Cited Sources")

    st.markdown("""
*   https://streamlit.io/
*   Author, A. (Year). *Title of the work*. Publisher/Website. [URL](http://example.com)
*   Author, B., & Author, C. (Year). *Article title*. Journal Name, Volume(Issue), pages. [URL](http://example.com)
*   Smith, J. (Year). *Presentation on Streamlit Layouts*. [https://example.com/presentation-link](https://example.com/presentation-link)
*   Streamlit Documentation. (n.d.). *Layouts and Containers*. [https://docs.streamlit.io/develop/api-reference/layout](https://docs.streamlit.io/develop/api-reference/layout)
    """)

    st.markdown("---")

    # --- Optional: Contact Information / Call to Action ---
    col1, col2 = st.columns(2)  # Use columns to neatly organize contact details or next steps

    with col1:
        st.subheader("Contact Information")
        st.markdown(f"*   **Presenter:** Ofer Tzvi")
        st.markdown(f"*   **Email:** ofer@il-fts.com")
        st.markdown(f"*   **LinkedIn:** [https://www.linkedin.com/in/ofertzvi/](linkedin.com)")

    with col2:
        st.subheader("Thank You!")
        st.markdown("*   Please feel free to reach out with any questions.")
        st.markdown("*   *This application was built using the [Streamlit library](https://streamlit.io).*")
#
# # --- Footer (Optional) ---
# st.sidebar.markdown("---")
# st.sidebar.markdown("© 2025 Your Company/Organization")