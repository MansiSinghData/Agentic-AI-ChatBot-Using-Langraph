import streamlit as st
import requests


st.set_page_config(page_title="Langraph Agentic AI", layout="centered")
st.title("Langraph Agentic AI Chatbot")
st.write("Create and Interact with the AI Agents.")

system_prompt=st.text_area("Define your AI Agent:", height=70, placeholder="Type your System Prompt here.")

Model_names_Groq=["llama-3.3-70b-versatile","mixtral-8x7b-32768"]
Model_names_OpenAI=["gpt-4o-mini"]

provider=st.radio("Select the model provider:", options=["Groq","OpenAI"])

if provider=="Groq":
    selected_model=st.selectbox("Select the model you want to use:", options=Model_names_Groq)
elif provider=="OpenAI":
    selected_model=st.selectbox("Select the model you want to use:", options=Model_names_OpenAI)



allow_search=st.checkbox("Allow Search Tool Access?")

query=st.text_area("Enter your query here:", height=100, placeholder="Ask Anything!")

api_url="http://127.0.0.1:9000/chat"

if st.button("Get Response"):
    if query.strip():
        payload={
            "model_name": selected_model,
            "model_provider": provider,
            "system_prompt": system_prompt,
            "messages": [query],
            "allow_search": allow_search
}
        
        response=requests.post(api_url, json=payload)
        if response.status_code==200:
            response_data=response.json()
            if "error" in response_data:
                st.error(response_data["error"])
            else:    
                st.subheader("Response from the Agent:")
                st.markdown(f"**Agent Says:** {response_data}")