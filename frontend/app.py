import streamlit as st
import requests

st.set_page_config(page_title="Telugu Spam Detector", page_icon="📩")

API_URL = "http://localhost:8000/predict"

st.title("📩 Telugu Spam SMS Detector")
st.markdown("Enter a Telugu SMS message below. The app uses a fine-tuned **MuRIL** model to detect Spam vs Ham.")

user_input = st.text_area("Enter Telugu SMS", height=150, placeholder="ఇక్కడ మీ SMS ని టైప్ చేయండి...")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        with st.spinner("Analyzing message..."):
            try:
                response = requests.post(API_URL, json={"text": user_input})
                if response.status_code == 200:
                    data = response.json()
                    label = data["label"]
                    confidence = data["confidence"]

                    if label == "Spam":
                        st.error(f"🚨 **Spam** (Confidence: {confidence:.2f})")
                    else:
                        st.success(f"✅ **Ham** (Confidence: {confidence:.2f})")
                else:
                    st.error(f"Server error: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")
