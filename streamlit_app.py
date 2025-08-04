# streamlit_app.py
import streamlit as st
from main import initialize_hindugpt, app
import os

# Page config
st.set_page_config(
    page_title="HinduGPT - Wisdom of the Gita",
    page_icon="🕉️",
    layout="centered"
)

# Title & Description
st.title("🕉️ HinduGPT")
st.markdown("""
> *“Whenever dharma declines, O Arjuna, I manifest Myself.”* — Bhagavad Gita 4.7
""")
st.markdown("Ask questions about **Krishna**, **Dharma**, **Karma**, **Life**, or anything — answered through the wisdom of the **Bhagavad Gita** or general knowledge.")

# Initialize HinduGPT only once
if "app" not in st.session_state:
    st.session_state.app = None
    st.session_state.messages = []
    st.session_state.initialized = False

# Lazy initialization
if not st.session_state.initialized:
    with st.spinner("🧠 Loading Bhagavad Gita and initializing AI... (This may take a minute)"):
        try:
            st.session_state.app = initialize_hindugpt()
            if st.session_state.app:
                st.session_state.initialized = True
                st.success("✅ HinduGPT is ready! Ask your question below.")
            else:
                st.error("💥 Failed to initialize. Please check logs and environment setup.")
        except Exception as e:
            st.error(f"💥 Initialization error: {str(e)}")
            st.info("💡 Make sure: \n- `data/Bhagavad_Gita_As_It_Is.pdf` exists \n- Your `.env` has valid API keys")

# Show chat only if initialized
if st.session_state.initialized:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about life, dharma, Krishna, or anything..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Seeking wisdom..."):
                try:
                    inputs = {"question": prompt}
                    result = None
                    for output in st.session_state.app.stream(inputs):
                        result = list(output.values())[0]

                    # Format response
                    docs = result["documents"]
                    if docs and isinstance(docs, list):
                        content = docs[0].page_content.strip()
                        if len(content) > 20:
                            response = f"📘 **From the Bhagavad Gita:**\n\n{content[:800]}..."
                        else:
                            response = f"🌐 **From Wikipedia:**\n\n{content}"
                    else:
                        content = docs[0].page_content if isinstance(docs, list) else getattr(docs, 'page_content', str(docs))
                        response = f"🌐 **General Knowledge:**\n\n{content[:800]}..."

                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = "I couldn't retrieve an answer right now. Please try again."
                    st.markdown(f"❌ {error_msg}")
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})