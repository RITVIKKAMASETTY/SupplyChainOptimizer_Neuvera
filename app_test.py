import streamlit as st

st.title("Streamlit Test App")
st.write("If you can see this, Streamlit is working correctly!")

slider_value = st.slider("Test slider", 0, 100, 50)
st.write(f"Selected value: {slider_value}")
