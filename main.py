import os
import warnings
from dotenv import load_dotenv
import cassio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from typing import Literal, List
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from typing_extensions import TypedDict
from langchain.schema import Document
from langgraph.graph import END, StateGraph, START
from pprint import pprint

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# Set USER_AGENT to avoid the warning
os.environ['USER_AGENT'] = 'Multi-Agentic-RAG-AI/1.0'

# Load environment variables
load_dotenv()

# Get API keys from environment
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print("🚀 Starting Multi-Agentic RAG AI...")

# Initialize Astra DB connection
print("📡 Connecting to Astra DB...")
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Build index
print("📚 Loading and processing documents...")
urls = [
    "https://www.prabhupada-books.de/pdf/Bhagavad-gita-As-It-Is.pdf"
]

# Load documents
docs = [WebBaseLoader(url).load() for url in urls]
doc_list = [item for sublist in docs for item in sublist]

# Split documents
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
docs_split = text_splitter.split_documents(doc_list)

# Create embeddings
print("🧠 Creating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# Set up vector store
print("💾 Setting up vector store...")
astra_vector_store = Cassandra(
    embedding=embeddings,
    table_name='qa_mini_demo',
    session=None,
    keyspace=None
)

# Add documents to vector store
astra_vector_store.add_documents(docs_split)
print(f'✅ Inserted {len(docs_split)} document chunks.')

# Create retriever
retriever = astra_vector_store.as_retriever()

# Data model for routing
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource"""
    datasource: Literal['vectorstore', 'wiki_search'] = Field(
        ...,
        description='Given a user question choose to route it to wikipedia or a vectorstore.'
    )

# Initialize LLM
print("🤖 Initializing language model...")
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name='Llama-3.3-70b-Versatile')
structured_llm_with_router = llm.with_structured_output(RouteQuery)

# Create router prompt
system = '''You are an expert at routing a user question to a vectorstore or wikipedia.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use wiki-search.'''

route_prompt = ChatPromptTemplate.from_messages([
    ('system', system),
    ('human', '{question}')
])
question_router = route_prompt | structured_llm_with_router

# Set up Wikipedia search
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# Graph state
class GraphState(TypedDict):
    """
    Represents the state of our graph
    
    Attributes:
        question: question
        generation: LLM generation  
        documents: list of documents
    """
    question: str
    generation: str
    documents: List[str]

# Node functions
def retrieve(state):
    """Retrieve documents"""
    print('🔍 RETRIEVING from Vector Store...')
    question = state['question']
    documents = retriever.invoke(question)
    return {'documents': documents, 'question': question}

def wiki_search(state):
    """Wiki search based on the question"""
    print('🌐 SEARCHING Wikipedia...')
    question = state['question']
    docs = wiki.invoke({'query': question})
    wiki_results = Document(page_content=docs)
    return {'documents': wiki_results, 'question': question}

def route_question(state):
    """Route question to wiki search or RAG"""
    print('🧭 ROUTING QUESTION...')
    question = state['question']
    source = question_router.invoke({'question': question})
    if source.datasource == 'wiki_search':
        print('   → Routing to WIKIPEDIA')
        return 'wiki_search'
    elif source.datasource == 'vectorstore':
        print('   → Routing to VECTOR STORE')
        return 'vectorstore'

# Build the graph
print("🔧 Building the workflow graph...")
workflow = StateGraph(GraphState)

# Define nodes
workflow.add_node('wiki_search', wiki_search)
workflow.add_node('retrieve', retrieve)

# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        'wiki_search': 'wiki_search',
        'vectorstore': 'retrieve'
    }
)

workflow.add_edge('retrieve', END)
workflow.add_edge('wiki_search', END)

# Compile
app = workflow.compile()

print("✅ Setup complete! Ready to answer questions!")
print("=" * 60)

def ask_question(question):
    """Ask a question to the multi-agentic system"""
    print(f"\n❓ QUESTION: {question}")
    print("-" * 50)
    
    inputs = {'question': question}
    
    result = None
    for output in app.stream(inputs):
        for key, value in output.items():
            result = value
            print(f'✅ Completed: {key.upper()}')
    
    print("-" * 50)
    
    # Print final result in a nicer format
    if isinstance(result['documents'], list):
        print("📄 RETRIEVED DOCUMENTS:")
        for i, doc in enumerate(result['documents'][:3], 1):  # Show first 3 docs
            print(f"\n📋 Document {i}:")
            content = doc.page_content.replace('\n', ' ')[:250] + "..."
            print(f"   {content}")
            if hasattr(doc, 'metadata') and doc.metadata.get('source'):
                print(f"   🔗 Source: {doc.metadata['source']}")
    else:
        print("📄 WIKIPEDIA RESULT:")
        content = result['documents'].page_content
        print(f"   {content}")
    
    print("=" * 60)

def interactive_mode():
    """Run in interactive mode"""
    print("\n🤖 INTERACTIVE MODE ACTIVATED!")
    print("💡 Try asking questions like:")
    print("   • What is an agent?")
    print("   • Who is Elon Musk?") 
    print("   • What is prompt engineering?")
    print("   • Tell me about machine learning")
    print("\n📝 Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        try:
            question = input("\n💭 Your question: ").strip()
            if question.lower() in ['quit', 'exit', 'bye', 'q']:
                print("👋 Thanks for using Multi-Agentic RAG AI!")
                break
            if question:
                ask_question(question)
        except KeyboardInterrupt:
            print("\n👋 Thanks for using Multi-Agentic RAG AI!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

# Example usage
if __name__ == "__main__":
    # Test with different types of questions
    print("🧪 RUNNING TEST QUESTIONS...")
    ask_question("What is an agent?")
    ask_question("Who is Shahrukh Khan?")
    ask_question("What is prompt engineering?")
    
    # Ask user if they want interactive mode
    print("\n" + "="*60)
    response = input("🎮 Would you like to try interactive mode? (y/n): ").lower()
    if response in ['y', 'yes']:
        interactive_mode()
    else:
        print("👋 Thanks for using Multi-Agentic RAG AI!")