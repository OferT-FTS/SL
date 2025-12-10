import base64

import streamlit as st
st.logo("combined_logo.png")

page_0 = st.Page("page_0.py", title = "ML DP Presentation")
page_1 = st.Page("page_1.py", title = "Problem Description")
page_2 = st.Page("page_2.py", title = "Research Data")
page_3 = st.Page("page_3.py", title = "Features Engineering and Selection")
page_4 = st.Page("page_4.py", title = "ML & DL Model Selection")
page_5 = st.Page("page_5.py", title = "Final Model Prediction")
page_6 = st.Page("page_6.py", title = "Behind The Scenes")
page_7 = st.Page("page_7.py", title = "Conclusions")
# page_8 = st.Page("page_8.py", title = "Python Project Overview")
# page_9 = st.Page("page_9.py", title = "Appreciation")

# Set up navigation
pg = st.navigation(
    [page_0, page_1,page_2, page_3, page_4, page_5, page_6, page_7]
)

# run the selected page
pg.run()

# 08-9170444
# 08-9268118
#