# Importing Libraries
import streamlit as st

performance_prediction_page = st.Page("performance_prediction.py", title = "Employee Performance Prediction", icon = ":material/online_prediction:")
data_analysis_page = st.Page("data_analysis.py", title = "Data Analysis", icon = ":material/chart_data:")
landing_page = st.Page("landing_page.py", title = "Home Page", icon = ":material/home:")


pg = st.navigation([landing_page,data_analysis_page, performance_prediction_page])
pg.run()
