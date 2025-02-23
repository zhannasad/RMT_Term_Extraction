import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Load the pre-trained BERT model and tokenizer
model_path = './bert_term_extraction'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')


# Function to clean and tokenize text
def clean_and_tokenize(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join([i for i in text if not i.isdigit()])
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return filtered_text


# Function to extract terms using the BERT model
def extract_terms_from_text(input_text):
    tokens = clean_and_tokenize(input_text)
    inputs = tokenizer(tokens, padding=True, truncation=True, max_length=128, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).numpy()

    terms = [tokens[i] for i in range(len(tokens)) if predictions[i] == 1]
    return terms


# Streamlit app
def main():
    st.title("Blockchain Term Extractor")
    st.write("This app extracts blockchain-related terms from the input text using a pre-trained BERT model.")

    # User input
    user_input = st.text_area("Enter the text you want to analyze:", height=150)

    if st.button("Extract Terms"):
        if user_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Extracting terms..."):
                extracted_terms = extract_terms_from_text(user_input)

            if extracted_terms:
                st.success("Extracted Terms:")
                st.write(", ".join(extracted_terms))
            else:
                st.info("No blockchain-related terms were found in the input text.")


if __name__ == "__main__":
    main()