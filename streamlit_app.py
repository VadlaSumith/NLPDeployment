import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned GPT-2 model and tokenizer
model_name = 'gpt2'  # Replace with your fine-tuned model if saved locally
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set the pad token to eos token
tokenizer.pad_token = tokenizer.eos_token

# Move the model to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Streamlit App Layout
st.title("GPT-2 Chatbot")
st.markdown("This app allows you to interact with a GPT-2 model that has been fine-tuned with Few-Shot and CoT examples.")

# Debug: Check if the app is running
st.write("Debug: App is running")

# Input Text Box
user_input = st.text_input("Ask something:", "")

# Check if user input is provided
if not user_input:
    st.write("Please type your question above and press Enter to get a response!")
else:
    st.write("Debug: User input received:", user_input)
    
    try:
        # Tokenize the input text and generate the response
        input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)
        
        # Generate a response using the model
        output = model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id)
        
        # Decode the generated response
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Display the response
        st.write("GPT-2 Response:")
        st.write(generated_text)
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Add more options like reset and customization if needed
