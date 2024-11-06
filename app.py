
import streamlit as st
from data_loading import *
from data_processing import *
from visualization import *

# Streamlit App Navigation
st.sidebar.title("Streamlit Multi-Page App")
page = st.sidebar.radio("Go to", ["Data Loading", "Data Processing", "Visualization"])

if page == "Data Loading":
    st.title("Data Loading")
    load_data()

elif page == "Data Processing":
    st.title("Data Processing")
    process_data()

elif page == "Visualization":
    st.title("Visualization")
    show_visualization()
