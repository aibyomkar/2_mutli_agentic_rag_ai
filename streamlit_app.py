import streamlit as st
from main import app

# Page config
st.set_page_config(
    page_title="Multi-Agentic RAG AI",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Multi-Agentic RAG AI")
st.markdown("Ask a question about AI, people, or tech.")

# Sidebar
with st.sidebar:
    if st.button("What is an AI agent?"):
        st.session_state.messages = [{"role": "user", "content": "What is an AI agent?"}]
    if st.button("🗑️ Clear Chat"):
        st.session_state.clear()

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle input
if prompt := st.chat_input("Your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking..."):
            try:
                state = None
                for chunk in app.stream({"question": prompt}):
                    state = list(chunk.values())[0]
                docs = state["documents"]
                content = docs[0] if isinstance(docs, list) else docs
                source = "📘 Vectorstore" if isinstance(docs, list) else "🌐 Wikipedia"
                response = f"**{source}**: {str(content)[:600]}..."
                st.markdown(response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
            except Exception as e:
                st.error(f"❌ {e}")