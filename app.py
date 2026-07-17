import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("spam_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Email Spam Detection", page_icon="📧", layout="centered")

st.title("📧 Email Spam Detection")
st.markdown("### Detect whether an email is **Spam** or **Not Spam**")

st.sidebar.header("About")
st.sidebar.write(
    """
    **Project:** Email Spam Detection
    
    **Model:** Logistic Regression
    
    Enter the email details and click **Predict**.
    """
)

email_id = st.number_input("Email ID", min_value=1, value=1)
sender_email = st.number_input("Sender Email (Encoded)", min_value=0, value=0)
subject = st.number_input("Subject (Encoded)", min_value=0, value=0)
email_length = st.number_input("Email Length", min_value=0, value=100)
num_links = st.number_input("Number of Links", min_value=0, value=1)
num_special_chars = st.number_input("Number of Special Characters", min_value=0, value=2)
capital_words = st.number_input("Capital Words", min_value=0, value=5)
has_attachment = st.selectbox("Has Attachment", [0, 1])

if st.button("🔍 Predict"):

    data = pd.DataFrame({
        "Email_ID": [email_id],
        "Sender_Email": [sender_email],
        "Subject": [subject],
        "Email_Length": [email_length],
        "Num_Links": [num_links],
        "Num_Special_Chars": [num_special_chars],
        "Capital_Words": [capital_words],
        "Has_Attachment": [has_attachment]
    })

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)
    probability = model.predict_proba(data_scaled)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("🚨 This Email is SPAM")
    else:
        st.success("✅ This Email is NOT SPAM")

    st.write(f"**Spam Probability:** {probability[0][1]*100:.2f}%")
    st.progress(float(probability[0][1]))

st.markdown("---")
st.caption("Developed using Python, Scikit-learn and Streamlit")