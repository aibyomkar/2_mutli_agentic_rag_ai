# streamlit_app.py
import streamlit as st
from main import initialize_hindugpt, app
import os

# Page config
st.set_page_config(
    page_title="🕉️ HinduGPT",
    page_icon="🕉️",
    layout="centered"
)

st.title("🕉️ HinduGPT")
st.markdown("### Your AI Guide to Bhagavad Gita & Sanatan Dharma")

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.messages = []

# Lazy initialization
if not st.session_state.initialized:
    with st.spinner("🧘 Initializing HinduGPT... (This may take a minute)"):
        try:
            initialize_hindugpt()
            st.session_state.initialized = True
            st.success("✅ HinduGPT is ready!")
        except Exception as e:
            st.error(f"❌ Initialization failed: {e}")
            st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
if prompt := st.chat_input("Ask a question about Krishna, Dharma, Gita, or anything..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Show thinking
    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking..."):
            try:
                inputs = {"question": prompt}
                result = None
                for output in app.stream(inputs):
                    result = list(output.values())[0]

                # Format response
                docs = result["documents"]
                if isinstance(docs, list) and len(docs) > 0:
                    content = docs[0].page_content
                    source = docs[0].metadata.get("source", "Bhagavad Gita") if hasattr(docs[0], 'metadata') else "Bhagavad Gita"
                    response = f"📘 Based on the Bhagavad Gita:\n\n{content[:800]}..."
                else:
                    content = docs[0].page_content if isinstance(docs, list) else docs.page_content
                    response = f"🌐 Wikipedia result:\n\n{content}"

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"❌ I couldn't process that. Error: {str(e)}"
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})