import streamlit as st
import pickle
import nltk
import string
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Set up NLTK directory inside project (works locally & on Streamlit Cloud)
NLTK_DIR = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(NLTK_DIR)
nltk.download('punkt', download_dir=NLTK_DIR, quiet=True)
nltk.download('stopwords', download_dir=NLTK_DIR, quiet=True)

ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Load saved model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩")
st.title("📩 Email / SMS Spam Classifier")
st.write("Paste your message below and check if it's **Spam** or **Not Spam**!")

input_sms = st.text_area("✍️ Enter the message here:")

if st.button('🔍 Predict'):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message before predicting.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Display Result
        if result == 1:
            st.error("🚨 This message is **SPAM**!")
        else:
            st.success("✅ This message is **NOT SPAM**.")
