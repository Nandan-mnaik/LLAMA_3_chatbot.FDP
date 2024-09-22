import streamlit as st
import replicate
import os

st.set_page_config(page_title="AI Chatbot")
# Sidebar for API token input and parameters
with st.sidebar:
    st.title('Llama 3.1 Chatbot')

    # Enter Replicate API token
    replicate_api = st.text_input('Enter Replicate API token:', type='password')

    # Model parameters
    temperature = st.slider('Temperature', min_value=0.01, max_value=1.0, value=0.5, step=0.01)
    top_p = st.slider('Top P', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.slider('Max Length', min_value=32, max_value=512, value=128, step=32)

# Store the Replicate API token in the environment variable if provided
if replicate_api:
    os.environ['REPLICATE_API_TOKEN'] = replicate_api

# Define the Replicate Llama 3.1 model version
llama_3_1_model = "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"

# Set up chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function to clear chat history
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function to generate response from the Replicate API
def generate_response(prompt_input):
    if not replicate_api:
        st.warning("Please enter a valid Replicate API token.")
        return None
    
    try:
        # API request to Replicate
        output = replicate.run(
            llama_3_1_model,
            input={
                "prompt": prompt_input,
                "temperature": temperature,
                "top_p": top_p,
                "max_length": max_length,
                "repetition_penalty": 1.0
            }
        )
        # Join the output response if it's a list (Replicate API often returns a list)
        return ''.join(output) if isinstance(output, list) else output
    except Exception as e:
        st.error(f"Error calling Replicate API: {e}")
        return None

# User input for chat
prompt = st.chat_input("Enter your message here...")

# Generate and display response if a prompt is entered
if prompt and replicate_api:
    # Add user input to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            if response:
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.warning("Failed to generate a response. Please try again.")
