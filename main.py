# import os
# import warnings
# from dotenv import load_dotenv
# import cassio
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Cassandra
# from langchain.indexes.vectorstore import VectorStoreIndexWrapper
# from typing import Literal, List
# from langchain_core.prompts import ChatPromptTemplate
# from pydantic import BaseModel, Field
# from langchain_groq import ChatGroq
# from langchain_community.utilities import WikipediaAPIWrapper
# from langchain_community.tools import WikipediaQueryRun
# from typing_extensions import TypedDict
# from langchain.schema import Document
# from langgraph.graph import END, StateGraph, START
# from pprint import pprint

# # Suppress warnings for cleaner output
# warnings.filterwarnings("ignore", category=FutureWarning)

# # Set USER_AGENT to avoid the warning
# os.environ['USER_AGENT'] = 'Multi-Agentic-RAG-AI/1.0'

# # Load environment variables
# load_dotenv()

# # Get API keys from environment
# ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
# ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# print("🚀 Starting Multi-Agentic RAG AI...")

# # Initialize Astra DB connection
# print("📡 Connecting to Astra DB...")
# cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# # Build index
# print("📚 Loading and processing documents...")
# urls = [
#     "https://www.prabhupada-books.de/pdf/Bhagavad-gita-As-It-Is.pdf"
# ]

# # Load documents
# docs = [WebBaseLoader(url).load() for url in urls]
# doc_list = [item for sublist in docs for item in sublist]

# # Split documents
# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=500, chunk_overlap=0
# )
# docs_split = text_splitter.split_documents(doc_list)

# # Create embeddings
# print("🧠 Creating embeddings...")
# embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# # Set up vector store
# print("💾 Setting up vector store...")
# astra_vector_store = Cassandra(
#     embedding=embeddings,
#     table_name='qa_mini_demo',
#     session=None,
#     keyspace=None
# )

# # Add documents to vector store
# astra_vector_store.add_documents(docs_split)
# print(f'✅ Inserted {len(docs_split)} document chunks.')

# # Create retriever
# retriever = astra_vector_store.as_retriever()

# # Data model for routing
# class RouteQuery(BaseModel):
#     """Route a user query to the most relevant datasource"""
#     datasource: Literal['vectorstore', 'wiki_search'] = Field(
#         ...,
#         description='Given a user question choose to route it to wikipedia or a vectorstore.'
#     )

# # Initialize LLM
# print("🤖 Initializing language model...")
# llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name='Llama-3.3-70b-Versatile')
# structured_llm_with_router = llm.with_structured_output(RouteQuery)

# # Create router prompt
# system = '''You are an expert at routing a user question to a vectorstore or wikipedia.
# The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
# Use the vectorstore for questions on these topics. Otherwise, use wiki-search.'''

# route_prompt = ChatPromptTemplate.from_messages([
#     ('system', system),
#     ('human', '{question}')
# ])
# question_router = route_prompt | structured_llm_with_router

# # Set up Wikipedia search
# api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
# wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# # Graph state
# class GraphState(TypedDict):
#     """
#     Represents the state of our graph
    
#     Attributes:
#         question: question
#         generation: LLM generation  
#         documents: list of documents
#     """
#     question: str
#     generation: str
#     documents: List[str]

# # Node functions
# def retrieve(state):
#     """Retrieve documents"""
#     print('🔍 RETRIEVING from Vector Store...')
#     question = state['question']
#     documents = retriever.invoke(question)
#     return {'documents': documents, 'question': question}

# def wiki_search(state):
#     """Wiki search based on the question"""
#     print('🌐 SEARCHING Wikipedia...')
#     question = state['question']
#     docs = wiki.invoke({'query': question})
#     wiki_results = Document(page_content=docs)
#     return {'documents': wiki_results, 'question': question}

# def route_question(state):
#     """Route question to wiki search or RAG"""
#     print('🧭 ROUTING QUESTION...')
#     question = state['question']
#     source = question_router.invoke({'question': question})
#     if source.datasource == 'wiki_search':
#         print('   → Routing to WIKIPEDIA')
#         return 'wiki_search'
#     elif source.datasource == 'vectorstore':
#         print('   → Routing to VECTOR STORE')
#         return 'vectorstore'

# # Build the graph
# print("🔧 Building the workflow graph...")
# workflow = StateGraph(GraphState)

# # Define nodes
# workflow.add_node('wiki_search', wiki_search)
# workflow.add_node('retrieve', retrieve)

# # Build graph
# workflow.add_conditional_edges(
#     START,
#     route_question,
#     {
#         'wiki_search': 'wiki_search',
#         'vectorstore': 'retrieve'
#     }
# )

# workflow.add_edge('retrieve', END)
# workflow.add_edge('wiki_search', END)

# # Compile
# app = workflow.compile()

# print("✅ Setup complete! Ready to answer questions!")
# print("=" * 60)

# def ask_question(question):
#     """Ask a question to the multi-agentic system"""
#     print(f"\n❓ QUESTION: {question}")
#     print("-" * 50)
    
#     inputs = {'question': question}
    
#     result = None
#     for output in app.stream(inputs):
#         for key, value in output.items():
#             result = value
#             print(f'✅ Completed: {key.upper()}')
    
#     print("-" * 50)
    
#     # Print final result in a nicer format
#     if isinstance(result['documents'], list):
#         print("📄 RETRIEVED DOCUMENTS:")
#         for i, doc in enumerate(result['documents'][:3], 1):  # Show first 3 docs
#             print(f"\n📋 Document {i}:")
#             content = doc.page_content.replace('\n', ' ')[:250] + "..."
#             print(f"   {content}")
#             if hasattr(doc, 'metadata') and doc.metadata.get('source'):
#                 print(f"   🔗 Source: {doc.metadata['source']}")
#     else:
#         print("📄 WIKIPEDIA RESULT:")
#         content = result['documents'].page_content
#         print(f"   {content}")
    
#     print("=" * 60)

# def interactive_mode():
#     """Run in interactive mode"""
#     print("\n🤖 INTERACTIVE MODE ACTIVATED!")
#     print("💡 Try asking questions like:")
#     print("   • What is an agent?")
#     print("   • Who is Elon Musk?") 
#     print("   • What is prompt engineering?")
#     print("   • Tell me about machine learning")
#     print("\n📝 Type 'quit' to exit")
#     print("=" * 60)
    
#     while True:
#         try:
#             question = input("\n💭 Your question: ").strip()
#             if question.lower() in ['quit', 'exit', 'bye', 'q']:
#                 print("👋 Thanks for using Multi-Agentic RAG AI!")
#                 break
#             if question:
#                 ask_question(question)
#         except KeyboardInterrupt:
#             print("\n👋 Thanks for using Multi-Agentic RAG AI!")
#             break
#         except Exception as e:
#             print(f"❌ Error: {e}")

# # Example usage
# if __name__ == "__main__":
#     # Test with different types of questions
#     print("🧪 RUNNING TEST QUESTIONS...")
#     ask_question("What is an agent?")
#     ask_question("Who is Shahrukh Khan?")
#     ask_question("What is prompt engineering?")
    
#     # Ask user if they want interactive mode
#     print("\n" + "="*60)
#     response = input("🎮 Would you like to try interactive mode? (y/n): ").lower()
#     if response in ['y', 'yes']:
#         interactive_mode()
#     else:
#         print("👋 Thanks for using Multi-Agentic RAG AI!")








































# # main.py
# import os
# import warnings
# from dotenv import load_dotenv
# import cassio
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Cassandra
# from typing import Literal, List
# from langchain_core.prompts import ChatPromptTemplate
# from pydantic import BaseModel, Field
# from langchain_groq import ChatGroq
# from langchain_community.utilities import WikipediaAPIWrapper
# from langchain_community.tools import WikipediaQueryRun
# from typing_extensions import TypedDict
# from langchain.schema import Document
# from langgraph.graph import END, StateGraph, START

# # Suppress warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# # Load environment variables
# load_dotenv()

# # API Keys
# ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
# ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# # Global variables
# embeddings = None
# retriever = None
# app = None

# def initialize_hindugpt():
#     """Initialize the HinduGPT system: load PDF from data/, create vector store, build graph."""
#     global retriever, app

#     print("🚀 Initializing HinduGPT...")

#     # Initialize Astra DB
#     if not ASTRA_DB_APPLICATION_TOKEN or not ASTRA_DB_ID:
#         print("❌ Astra DB credentials missing! Check your .env file.")
#         return None

#     print("📡 Connecting to Astra DB...")
#     try:
#         cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
#     except Exception as e:
#         print(f"❌ Failed to connect to Astra DB: {e}")
#         return None

#     # Embeddings
#     print("🧠 Loading embeddings model...")
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#     # Load Bhagavad Gita PDF from data folder
#     pdf_path = os.path.join("data", "Bhagavad_Gita_As_It_Is.pdf")
#     print(f"📚 Looking for PDF at: {os.path.abspath(pdf_path)}")

#     if not os.path.exists(pdf_path):
#         print(f"❌ PDF not found at {pdf_path}")
#         print("Please make sure you have a 'data' folder with 'Bhagavad_Gita_As_It_Is.pdf' inside.")
#         return None

#     try:
#         print("📖 Loading Bhagavad Gita PDF...")
#         loader = PyPDFLoader(pdf_path)
#         doc_list = loader.load()
#         print(f"✅ Loaded {len(doc_list)} pages from the Gita.")
#     except Exception as e:
#         print(f"❌ Failed to load PDF: {e}")
#         return None

#     # Split text
#     print("✂️  Splitting documents into chunks...")
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     docs_split = text_splitter.split_documents(doc_list)

#     # Vector Store
#     print("💾 Creating vector store in Astra DB...")
#     try:
#         astra_vector_store = Cassandra(
#             embedding=embeddings,
#             table_name="hindugpt_geeta",
#             session=None,
#             keyspace=None
#         )
#         astra_vector_store.add_documents(docs_split)
#         print(f"✅ Inserted {len(docs_split)} document chunks into vector store.")
#     except Exception as e:
#         print(f"❌ Failed to create vector store: {e}")
#         return None

#     retriever = astra_vector_store.as_retriever()

#     # LLM
#     if not GROQ_API_KEY:
#         print("❌ GROQ_API_KEY is missing! Please set it in .env")
#         return None

#     print("🤖 Initializing ChatGroq LLM...")
#     llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama-3.3-70b-Versatile")

#     # Router model
#     class RouteQuery(BaseModel):
#         datasource: Literal["vectorstore", "wiki_search"] = Field(
#             ..., description="Route to vectorstore (Geeta) or wiki"
#         )

#     structured_llm_router = llm.with_structured_output(RouteQuery)

#     route_prompt = ChatPromptTemplate.from_messages([
#         ("system", """
#         You are an expert router for HinduGPT.
#         Route queries about Bhagavad Gita, Sanatan Dharma, Krishna, Dharma, Karma, Arjuna, etc. to 'vectorstore'.
#         All other general knowledge questions go to 'wiki_search'.
#         """),
#         ("human", "{question}")
#     ])
#     question_router = route_prompt | structured_llm_router

#     # Wikipedia tool
#     wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
#     wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

#     # State
#     class GraphState(TypedDict):
#         question: str
#         generation: str
#         documents: List[Document]

#     # Nodes
#     def retrieve(state):
#         print("🔍 Retrieving from Bhagavad Gita vector store...")
#         docs = retriever.invoke(state["question"])
#         return {"documents": docs, "question": state["question"]}

#     def wiki_search(state):
#         print("🌐 Searching Wikipedia...")
#         try:
#             result = wiki_tool.invoke({"query": state["question"]})
#             doc = Document(page_content=result)
#             return {"documents": [doc], "question": state["question"]}
#         except Exception as e:
#             error_doc = Document(page_content=f"⚠️ Wikipedia search failed: {str(e)}")
#             return {"documents": [error_doc], "question": state["question"]}

#     def route_question(state):
#         print("🧭 Routing question...")
#         try:
#             source = question_router.invoke({"question": state["question"]})
#             return source.datasource
#         except Exception:
#             # Fallback to wiki if LLM fails
#             return "wiki_search"

#     # Build Graph
#     workflow = StateGraph(GraphState)
#     workflow.add_node("retrieve", retrieve)
#     workflow.add_node("wiki_search", wiki_search)

#     workflow.add_conditional_edges(
#         START,
#         route_question,
#         {
#             "vectorstore": "retrieve",
#             "wiki_search": "wiki_search"
#         }
#     )
#     workflow.add_edge("retrieve", END)
#     workflow.add_edge("wiki_search", END)

#     app = workflow.compile()
#     print("✅ HinduGPT initialization complete!")
#     return app





































import os
import warnings
from dotenv import load_dotenv
import cassio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Cassandra
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from typing_extensions import TypedDict
from langchain.schema import Document
from langgraph.graph import END, StateGraph, START

warnings.filterwarnings("ignore")
load_dotenv()

# Global variables
retriever, app = None, None

class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "wiki_search"] = Field(
        description="Route to vectorstore for Gita/Hindu topics, wiki for general knowledge"
    )

class GraphState(TypedDict):
    question: str
    documents: list

def check_data_exists(vector_store):
    """Check if Gita data already exists"""
    try:
        return bool(vector_store.similarity_search("Krishna", k=1))
    except:
        return False

def setup_vector_store():
    """Setup vector store - load PDF only if not already present"""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = Cassandra(
        embedding=embeddings,
        table_name="hindugpt_geeta",
        session=None,
        keyspace=None
    )
    
    if check_data_exists(vector_store):
        print("✅ Bhagavad Gita already loaded!")
        return vector_store.as_retriever()
    
    print("📖 Loading Bhagavad Gita for first time...")
    pdf_path = os.path.join("data", "Bhagavad_Gita_As_It_Is.pdf")
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at {pdf_path}")
    
    # Load and split PDF
    docs = PyPDFLoader(pdf_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    
    # Upload to vector store
    vector_store.add_documents(chunks)
    print(f"✅ Uploaded {len(chunks)} chunks to vector store!")
    
    return vector_store.as_retriever()

def initialize():
    """Initialize HinduGPT system"""
    global retriever, app
    
    print("🚀 Initializing HinduGPT...")
    
    # Validate credentials
    if not all([os.getenv("ASTRA_DB_APPLICATION_TOKEN"), os.getenv("ASTRA_DB_ID"), os.getenv("GROQ_API_KEY")]):
        raise ValueError("❌ Missing credentials in .env file!")
    
    # Initialize Astra DB
    cassio.init(token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"), database_id=os.getenv("ASTRA_DB_ID"))
    
    # Setup retriever
    retriever = setup_vector_store()
    
    # Setup LLM and router
    llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="Llama-3.3-70b-Versatile")
    router = ChatPromptTemplate.from_messages([
        ("system", "Route Bhagavad Gita/Hindu philosophy questions to 'vectorstore', others to 'wiki_search'"),
        ("human", "{question}")
    ]) | llm.with_structured_output(RouteQuery)
    
    # Setup Wikipedia
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300))
    
    # Define nodes
    def retrieve(state):
        print("🔍 Searching Bhagavad Gita...")
        return {"documents": retriever.invoke(state["question"]), "question": state["question"]}
    
    def wiki_search(state):
        print("🌐 Searching Wikipedia...")
        try:
            result = wiki.invoke({"query": state["question"]})
            return {"documents": [Document(page_content=result)], "question": state["question"]}
        except Exception as e:
            return {"documents": [Document(page_content=f"Wikipedia search failed: {e}")], "question": state["question"]}
    
    def route_question(state):
        try:
            return router.invoke({"question": state["question"]}).datasource
        except:
            return "wiki_search"
    
    # Build workflow
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("wiki_search", wiki_search)
    workflow.add_conditional_edges(START, route_question, {"vectorstore": "retrieve", "wiki_search": "wiki_search"})
    workflow.add_edge("retrieve", END)
    workflow.add_edge("wiki_search", END)
    
    app = workflow.compile()
    print("✅ HinduGPT ready!")

def ask(question):
    """Ask a question to HinduGPT"""
    if not app:
        initialize()
    
    print(f"\n🙏 Q: {question}")
    print("-" * 50)
    
    result = None
    for output in app.stream({"question": question}):
        for key, value in output.items():
            result = value
    
    # Display results
    if result and "documents" in result:
        docs = result["documents"]
        if docs:
            print("📚 Answer:")
            for i, doc in enumerate(docs[:2], 1):
                content = doc.page_content.replace('\n', ' ').strip()[:250] + "..."
                print(f"{i}. {content}")
        else:
            print("⚠️ No results found")
    
    print("=" * 50)
    return result

def chat():
    """Interactive chat mode"""
    print("🕉️ HinduGPT Chat Mode")
    print("Ask about Bhagavad Gita or general knowledge. Type 'quit' to exit.\n")
    
    while True:
        try:
            question = input("🙏 You: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                print("🙏 Namaste!")
                break
            if question:
                ask(question)
        except KeyboardInterrupt:
            print("\n🙏 Namaste!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Test questions
    ask("What does Krishna say about duty?")
    ask("Who is Albert Einstein?")
    
    # Interactive mode
    response = input("\n💬 Start chat mode? (y/n): ")
    if response.lower() in ['y', 'yes']:
        chat()