import os
import torch
import torch.nn as nn
import torch.optim as optim
import google.generativeai as genai
import streamlit as st
from transformers import GPT2Tokenizer
from pathlib import Path
from api_key import api_key
import re


# Gemini API Configuration

genai.configure(api_key=api_key)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

gemini_model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
)


# Data Cleaning Function

def clean_text(text):
    """Removes special characters and extra spaces."""
    text = re.sub(r"[^a-zA-Z0-9.,?!\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Tokenization Function

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_text(text):
    """Encodes text using GPT2 tokenizer."""
    return tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")


# Positional Encoding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


# Transformer Model with Attention and Add&Norm

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, ff_hidden_dim):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ff_hidden_dim)
        self.transformer = nn.TransformerEncoder(self.encoder_layers, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding(x)
        x = self.transformer(x)
        return self.fc_out(x)


# Streamlit UI for User Interaction

st.set_page_config(page_title="Disease Diagnosis", page_icon=":pill:")
st.image("ai-in-healthcare-icon-in-illustration-vector.jpg", width=150)
st.title("👩‍⚕Vital ♥ Image 📷 Analytics 📈👨‍⚕")
st.subheader("Upload a text file or medical image for analysis")

uploaded_file = st.file_uploader("Upload a text file (symptoms) or medical image", type=["txt", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_type = uploaded_file.type

    if "text" in file_type:
        # Handle text file
        symptoms_text = uploaded_file.read().decode("utf-8")
        symptoms_text = clean_text(symptoms_text)
        st.text_area("Symptoms from the file:", value=symptoms_text, height=150)

        if st.button("Analyze Symptoms"):
            # Use Gemini to analyze symptoms
            chat_history = [
                {
                    "role": "user",
                    "parts": [
                        f"I have the following symptoms: {symptoms_text}. What is the disease and what are the treatments?",
                    ],
                }
            ]

            chat_session = gemini_model.start_chat(history=chat_history)
            response = chat_session.send_message("Analyze the symptoms and provide a diagnosis and treatment.")

            st.subheader("Analysis Results:")
            st.write(response.text)

            # Tokenize and process the response using the Transformer model
            tokens = tokenize_text(response.text)
            st.write(f"Tokens: {tokens}")

            transformer_model = SimpleTransformer(
                vocab_size=len(tokenizer), d_model=512, num_heads=8, num_layers=6, ff_hidden_dim=2048
            )
            transformer_model.eval()

            with torch.no_grad():
                output = transformer_model(tokens)

            if output is not None and len(output.shape) > 1:
                try:
                    output_text = tokenizer.decode(output[0, -1].tolist(), skip_special_tokens=True)
                    st.subheader("Model Output (Transformer):")
                    st.write(output_text)
                except Exception as e:
                    st.error(f"Error decoding the output: {e}")
            else:
                st.error("Model output is empty or invalid.")

    elif "image" in file_type:
        # Handle medical image
        st.image(uploaded_file, caption="Uploaded Medical Image", use_column_width=True)

        if st.button("Analyze Image"):
            # Use Gemini to analyze medical image
            chat_history = [
                {
                    "role": "user",
                    "parts": [
                        "I have uploaded a medical image. Can you analyze it and provide insights?",
                    ],
                }
            ]

            chat_session = gemini_model.start_chat(history=chat_history)
            response = chat_session.send_message("Analyze the medical image and provide diagnosis and treatment suggestions.")

            st.subheader("Image Analysis Results:")
            st.write(response.text)
    else:
        st.error("Unsupported file type. Please upload a text file or an image.")
