# streamlit_app.py
import streamlit as st
from main import initialize, app
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
> *"Whenever dharma declines, O Arjuna, I manifest Myself."* — Bhagavad Gita 4.7
""")
st.markdown("Ask questions about **Krishna**, **Dharma**, **Karma**, **Life**, or anything — answered through the wisdom of the **Bhagavad Gita** or general knowledge.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.initialized = False

# One-time initialization
if not st.session_state.initialized:
    with st.spinner("🧠 Loading Bhagavad Gita... (First time may take a minute)"):
        try:
            initialize()
            st.session_state.initialized = True
            st.success("✅ HinduGPT is ready!")
        except Exception as e:
            st.error(f"💥 Error: {str(e)}")
            st.info("💡 Check: data/Bhagavad_Gita_As_It_Is.pdf exists and .env has valid keys")
            st.stop()

# Chat interface
if st.session_state.initialized:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about life, dharma, Krishna, or anything..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Seeking wisdom..."):
                try:
                    result = None
                    for output in app.stream({"question": prompt}):
                        result = list(output.values())[0]

                    # Format response
                    if result and "documents" in result:
                        docs = result["documents"]
                        if docs:
                            content = docs[0].page_content.strip()
                            
                            # Determine source and format response
                            if "Krishna" in content or "Arjuna" in content or "dharma" in content.lower():
                                response = f"📘 **From the Bhagavad Gita:**\n\n{content[:600]}..."
                            else:
                                response = f"🌐 **General Knowledge:**\n\n{content[:600]}..."
                        else:
                            response = "⚠️ No relevant information found."
                    else:
                        response = "❌ Unable to process your question. Please try again."

                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar with info
with st.sidebar:
    st.markdown("### 🕉️ About HinduGPT")
    st.markdown("Spiritual AI powered by the **Bhagavad Gita**")
    
    st.markdown("### 💫 Try asking:")
    st.markdown("""
    - What does Krishna say about duty?
    - How to overcome fear?
    - What is karma yoga?
    - Who is Elon Musk? (general knowledge)
    """)
    
    if st.button("🔄 Clear Chat"):
        st.session_state.messages = []
        st.rerun()