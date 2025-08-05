# streamlit_app.py
import streamlit as st
from main import app, initialize, ask_question  # assuming you add initialize() wrapper
import os

# ======================
# Page Configuration
# ======================
st.set_page_config(
    page_title="Multi-Agentic RAG AI",
    page_icon="🤖",
    layout="centered"
)

# ======================
# App Title & Description
# ======================
st.title("🤖 Multi-Agentic RAG AI")
st.markdown("""
> *"Route intelligently, retrieve wisely."*

A smart AI system that **routes your question** to the best source:
- 📚 **Vectorstore**: For topics like *agents, prompt engineering, adversarial attacks*
- 🌐 **Wikipedia**: For general knowledge (e.g., people, events, concepts)
""")

# ======================
# Sidebar Info Panel
# ======================
with st.sidebar:
    st.header("🧠 About This AI")
    st.markdown("""
    This app uses:
    - **LangGraph** for workflow routing
    - **Astra DB + Cassandra** for vector storage
    - **Groq (Llama-3.3-70b)** for reasoning
    - **Hugging Face Embeddings** for semantic search
    - **Wikipedia API** for open-domain facts
    """)

    st.markdown("### 🧩 Try These Examples:")
    sample_questions = [
        "What is an AI agent?",
        "Who is Elon Musk?",
        "Explain prompt engineering.",
        "What are adversarial attacks in AI?",
        "Tell me about India's space program"
    ]
    for q in sample_questions:
        if st.button(f"💬 {q}", key=q):
            st.session_state.messages = []
            st.session_state.pending_question = q

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ======================
# Initialize System Once
# ======================
@st.cache_resource
def load_system():
    from main import app  # ensure app is compiled
    print("✅ Multi-Agentic RAG system loaded via cache.")
    return app

# Load the app workflow
try:
    _ = load_system()
except Exception as e:
    st.error(f"💥 Failed to initialize AI system: {e}")
    st.stop()

# ======================
# Session State for Chat
# ======================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "source" in message:
            with st.expander("📄 Retrieved Content"):
                st.text(message["source"])

# ======================
# Handle Input
# ======================
user_input = st.chat_input("Ask a question about AI, people, tech, or anything...")

# Use pending question (from button) or user input
question = st.session_state.pending_question or user_input

if question:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Show thinking animation
    with st.chat_message("assistant"):
        with st.spinner("🧭 Routing your question..."):
            try:
                inputs = {"question": question}
                result = None
                final_state = None

                # Stream through the graph
                for output in app.stream(inputs):
                    final_state = output
                    step = list(output.keys())[0]
                    st.write(f"✅ {step.capitalize()} completed.")

                if final_state:
                    result = list(final_state.values())[-1]
                    docs = result['documents']
                    source_name = ""
                    content_preview = ""

                    # Format response based on source
                    if isinstance(docs, list) and len(docs) > 0:
                        # Came from vectorstore
                        doc = docs[0]
                        content = doc.page_content.replace('\n', ' ').strip()
                        source_name = "📘 From Internal Knowledge (Vectorstore)"
                        content_preview = f"{content[:600]}..."
                        full_content = content

                        if hasattr(doc, 'metadata') and doc.metadata.get('source'):
                            content_preview += f"\n\n🔗 Source: {doc.metadata['source']}"

                    elif isinstance(docs, str):
                        # Wikipedia result
                        content_preview = docs.strip()
                        full_content = content_preview
                        source_name = "🌐 From Wikipedia"
                    else:
                        content_preview = "No content retrieved."
                        full_content = content_preview
                        source_name = "⚠️ No Information Found"

                    # Display response
                    st.markdown(f"**{source_name}**")
                    st.markdown(content_preview)

                    # Save assistant response with expandable source
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"{source_name}\n\n{content_preview}",
                        "source": full_content
                    })

            except Exception as e:
                error_msg = f"❌ Error processing question: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

    # Reset pending question
    st.session_state.pending_question = None