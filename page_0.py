import streamlit as st

# Spacing
st.markdown("<br>", unsafe_allow_html=True)

st.set_page_config(
    page_title="Final Data Science Project",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

# Header with emoji
st.markdown("""
    <h1 style='text-align: center; color: gold; font-weight: 350'>
        Final Data Science Project - BIU DS20
    </h1>
""", unsafe_allow_html=True)

st.sidebar.markdown("## ML DP Presentation")

# Spacing
st.markdown("<br>", unsafe_allow_html=True)
# Spacing
st.markdown("<br>", unsafe_allow_html=True)
# Spacing
st.markdown("<br>", unsafe_allow_html=True)
# Spacing
st.markdown("<br>", unsafe_allow_html=True)
# Two symmetrical columns
col1, col2 = st.columns([1, 1])

# ===== LEFT SIDE =====
with col1:
    st.markdown("""
        <div style='text-align: center;'>
            <h2>Presenter</h2>
            <p style='font-size:20px; color: DimGray;'>Ofer Tzvi</p>
            <h2 style='color:#f0f2f6; text-align: center'>Project Goals</h2>
            <ul style='font-size:18px; color: DimGray; list-style-type: none; padding-left: 0; text-align: center;'>
                <li>Problem Solving with ML/DL Algo's</li>
                <li>Understanding Technologies Involved</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)


# ===== RIGHT SIDE =====
with col2:
    st.markdown("""
        <div style='text-align: center;'>
            <h2>Presentation Time</h2>
            <p style='font-size:17px; color: DimGray'>~ 20 minutes</p>
            <h2 style='text-align:center'>Target Audience</h2>
            <ul style='font-size:18px; color: DimGray; list-style-type: none; padding-left: 0;'>
                <li>DS Students</li>
                <li>Teaching Staff 🙏</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# import streamlit as st
#
# # from streamlit_app import logo
#
# # Page title
# # st.markdown("<h1 style='text-align: center;'>Final Data Science Project</h1>", unsafe_allow_html=True)
# # st.markdown("""
# #     <h1 style='text-align: center; color:gold;font-weight:700'>
# #         Final Data Science Project
# #     </h1>
# # """, unsafe_allow_html=True)
# st.set_page_config(
#     page_title="Final Data Science Project",
#     page_icon=":chart_with_upwards_trend:",  # 🡅 small chart emoji
#     layout="wide"
# )
#
# # Now your header
# st.markdown("""
#     <h1 style='text-align: center; color: gold; font-weight: 700'>
#         Final Data Science Project
#     </h1>
# """, unsafe_allow_html=True)
# st.sidebar.markdown("## ML DP Presentation")
#
# # Create centered main container
# st.markdown("<br>", unsafe_allow_html=True)
# col1, col2 = st.columns([3 , 3])
#
# # ===== LEFT SIDE =====
# with col1:
#     st.markdown("""
#         <div style='text-align: left;'>
#             <h2>Presenter</h2>
#             <p style='font-size:20px; font-weight:600; color: DimGray;'>Ofer Tzvi</p>
#             <h2 style='color:#f0f2f6'>Project Goals</h2>
#             <ul style='text-align: left; font-size:18px;; color: DimGray'>
#                 <li>Models Understanding</li>
#                 <li>Models Implementation</li>
#                 <li>Project's Technologies</li>
#             </ul>
#         </div>
#     """, unsafe_allow_html=True)
# #
# # with col2:
# #     st.markdown(" ")
# # ===== RIGHT SIDE =====
# with col2:
#     st.markdown("""
#         <div style='text-align: left;'>
#             <h2>Presentation Time</h2>
#             <p style='font-size:20px; font-weight:600; color: DimGray'>~ 20 minutes</p>
#             <h2>Presentation Audience</h2>
#             <ul style='text-align: left; font-size:18px; color: DimGray'>
#                 <li>Data science students</li>
#                 <li>Academic teaching staff</li>
#             </ul>
#         </div>
#     """, unsafe_allow_html=True)
#
#
# # # import streamlit as st
# # #
# # # # ===== MAKE STREAMLIT CONTENT FULL WIDTH =====
# # # st.markdown("""
# # #     <style>
# # #         /* Unlock full width for the main container */
# # #         .block-container {
# # #             padding-top: 0px;
# # #             padding-left: 0;
# # #             padding-right: 0;
# # #             max-width: 100% !important;
# # #         }
# # #     </style>
# # # """, unsafe_allow_html=True)
# # #
# # # # ===== FULL-WIDTH HEADER =====
# # # st.markdown("""
# # #     <div style="
# # #         width: 100vw;                /* Full viewport width */
# # #         position: relative;
# # #         left: 50%;
# # #         right: 50%;
# # #         margin-left: -50vw;          /* Pull to full width */
# # #         margin-right: -50vw;
# # #         background-color: #8B0000;   /* Change color */
# # #         padding: 25px 0;
# # #         text-align: center;
# # #         font-size: 40px;
# # #         font-weight: bold;
# # #     ">
# # #         Final Data Science Project
# # #     </div>
# # # """, unsafe_allow_html=True)
# # #
# # # # ===== SIDEBAR =====
# # # st.sidebar.markdown("## ML DP Presentation")
# # #
# # # # Spacing
# # # st.markdown("<br>", unsafe_allow_html=True)
# # #
# # # # Two columns
# # # col1, col2 = st.columns([1, 1])
# # #
# # # # ===== LEFT =====
# # # with col1:
# # #     st.markdown("""
# # #         <h2>Presenter</h2>
# # #         <p style='font-size:20px; font-weight:600;'>Ofer Tzvi</p>
# # #
# # #         <h3>Project Goals</h3>
# # #         <ul style='font-size:18px;'>
# # #             <li>Machine and Deep learning Models Understanding</li>
# # #             <li>Machine and Deep learning Models Implementation</li>
# # #             <li>Project's Technologies</li>
# # #         </ul>
# # #     """, unsafe_allow_html=True)
# # #
# # # # ===== RIGHT =====
# # # with col2:
# # #     st.markdown("""
# # #         <h2>Presentation Time</h2>
# # #         <p style='font-size:20px; font-weight:600;'>~ 20 minutes</p>
# # #
# # #         <h3>Presentation Audience</h3>
# # #         <ul style='font-size:18px;'>
# # #             <li>Data science students</li>
# # #             <li>Teaching staff</li>
# # #         </ul>
# # #     """, unsafe_allow_html=True)

