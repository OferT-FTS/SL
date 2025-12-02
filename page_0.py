import streamlit as st

# from streamlit_app import logo

# Page title
st.markdown("<h1 style='text-align: center;'>Final Data Science Project</h1>", unsafe_allow_html=True)

st.sidebar.markdown("## ML DP Presentation")

# Create centered main container
st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])

# ===== LEFT SIDE =====
with col1:
    st.markdown("""
        <div style='text-align: left;'>
            <h2>Presenter</h2>
            <p style='font-size:20px; font-weight:600;'>Ofer Tzvi</p>
            <h3>Project Goals</h3>
            <ul style='text-align: left; font-size:18px;'>
                <li>Machine and Deep learning Models Understanding</li>
                <li>Machine and Deep learning Models Implementation</li>
                <li>Project's Technologies</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# ===== RIGHT SIDE =====
with col2:
    st.markdown("""
        <div style='text-align: left;'>
            <h2>Presentation Time</h2>
            <p style='font-size:20px; font-weight:600;'>~ 20 minutes</p>
            <h3>Presentation Audience</h3>
            <ul style='text-align: left; font-size:18px;'>
                <li>Data science students</li>
                <li>Teaching staff</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)


