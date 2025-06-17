import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-cybercrime")
    model = GPT2LMHeadModel.from_pretrained("gpt2-cybercrime")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.title("🛡️ Cybercrime Detection & Mitigation GPT-2")

user_input = st.text_area("📝 Enter a prompt (e.g., a scenario or cybercrime-related question):", height=200)

if st.button("🔍 Generate Response"):
    if user_input.strip():
        inputs = tokenizer.encode(user_input, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.success(response)
    else:
        st.warning("Please enter a prompt.")
