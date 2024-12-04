import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load Models
model_paths = {
    "Base Model": "./new_trained_model",  # Update with the path to your first model
    "New Model 2": "./new2_trained_model",  # Update with the path to your second model
}

# UI for Model Selection
st.title("Fine-tuned GPT-2 Chatbot")
selected_model_name = st.selectbox("Select a model to use:", list(model_paths.keys()))

# Load the selected model and tokenizer
selected_model_path = model_paths[selected_model_name]
tokenizer = GPT2Tokenizer.from_pretrained(selected_model_path)
model = GPT2LMHeadModel.from_pretrained(selected_model_path)

st.write(f"### You are using: **{selected_model_name}**")

# Input Text
st.write("### Enter your message:")
input_text = st.text_input("Your message:", placeholder="Type something here...")

# Generate Response Button
if st.button("Generate Response"):
    if input_text.strip():  # Check for non-empty input
        # Generate response
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
        outputs = model.generate(
            **inputs,
            max_length=50,  # Adjust maximum response length to be shorter
            temperature=0.7,  # Add some randomness
            top_k=50,         # Limit choices to top 50
            top_p=0.95,       # Nucleus sampling
            repetition_penalty=1.2,  # Penalize repetition
            no_repeat_ngram_size=2,  # Avoid repeated n-grams
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Display the Response
        st.write("### Response:")
        st.text_area("Chatbot Response:", response, height=100)
    else:
        st.warning("Please enter a message to get a response.")

# Footer
st.markdown("---")
st.write("Developed with 💻 using [Streamlit](https://streamlit.io/) and [Hugging Face](https://huggingface.co/).")