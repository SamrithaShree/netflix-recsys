import streamlit as st
from src.api.model import load_model, recommend  # Adjust if your ML logic is elsewhere

# Load model (adjust path/imports as necessary)
model = load_model()

st.title("Netflix Recommendation System")

user_id = st.text_input("Enter User ID")

if st.button("Get Recommendations"):
    if user_id:
        recommendations = recommend(model, user_id)  # Your existing ML recommendation function
        st.write("Recommended Shows:")
        for show in recommendations:
            st.write(f"- {show}")
    else:
        st.error("Please enter a valid User ID.")
