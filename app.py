import streamlit as st
import os
import time
from typing import List, Dict, TypedDict, Optional

# --- LangChain Core Components ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document

# --- LLMs and Rerankers ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import CohereRerank

# --- Vector Store and Embeddings ---
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# --- Graph (Agent) Components ---
from langgraph.graph import StateGraph, END

# --- Constants ---
DATA_DIR = "data"
DB_DIR = "db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Note: Use 'mps' for Apple Silicon, 'cuda' for NVIDIA, or 'cpu'
# 'mps' will be very fast on your M4
EMBEDDING_DEVICE = "cpu" 
RETRIEVER_K = 10  # Number of docs to retrieve
RERANKER_TOP_N = 3  # Number of docs to pass to LLM

# --- 1. Caching and Resource Loading (Essential for Streamlit) ---
# These functions will only run ONCE, loading our models into memory.

@st.cache_resource
def get_llm():
    """Load the Google Gemini LLM."""
    try:
        # Get the API key from Streamlit secrets
        api_key = st.secrets["GOOGLE_API_KEY"]
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-09-2025", 
            google_api_key=api_key, 
            streaming=True
        )
    except KeyError:
        st.error("GOOGLE_API_KEY not found in secrets. Please add it to .streamlit/secrets.toml")
        st.stop()
    except Exception as e:
        st.error(f"Error loading Gemini LLM: {e}")
        st.stop()

@st.cache_resource
def get_reranker():
    """Load the Cohere Reranker."""
    try:
        api_key = st.secrets["COHERE_API_KEY"]
        return CohereRerank(model="rerank-english-v3.0", top_n=RERANKER_TOP_N, cohere_api_key=api_key)
    except KeyError:
        st.error("COHERE_API_KEY not found in secrets. Please add it to .streamlit/secrets.toml")
        st.stop()
    except Exception as e:
        st.error(f"Error loading Cohere Reranker: {e}")
        st.stop()

@st.cache_resource
def get_embedding_function():
    """Load the local, open-source embedding model."""
    try:
        return SentenceTransformerEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': EMBEDDING_DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        st.error(f"Error loading embedding model (Are 'sentence-transformers' and 'torch' installed?): {e}")
        st.stop()

@st.cache_resource
def get_retriever(_embeddings):
    """Load the ChromaDB vector store retriever from the 'db' folder."""
    if not os.path.exists(DB_DIR):
        st.error(f"Vector store not found in '{DB_DIR}'. Did you run 'ingest.py' first?")
        st.stop()
    
    try:
        vector_store = Chroma(
            persist_directory=DB_DIR,
            embedding_function=_embeddings
        )
        # Load the retriever
        return vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        st.stop()

@st.cache_data
def get_policy_info():
    """
    Dynamically get policy titles and numbers from the filenames in the /data dir.
    This is used to build the "intent detection" prompt.
    """
    policy_info = []
    try:
        for filename in os.listdir(DATA_DIR):
            if filename.endswith(".txt"):
                base_name = os.path.splitext(filename)[0]
                parts = base_name.split('_', 1)
                if len(parts) == 2:
                    policy_info.append({
                        "number": parts[0],
                        "title": parts[1].replace('_', ' ')
                    })
        return policy_info
    except FileNotFoundError:
        st.error(f"Data directory '{DATA_DIR}' not found. Please create it and add policy files.")
        st.stop()

# --- 2. LangGraph Agent Definition ---

class AgentState(TypedDict):
    """
    This class defines the "state" of our agent.
    It's a dictionary that holds all the data as it moves through the graph.
    """
    question: str
    policy_intent: str
    multi_queries: List[str]
    documents: List[Document]
    reranked_documents: List[Document]
    answer: str
    follow_up_questions: List[str]
    metrics: Dict[str, float]
    # Keep track of intermediate steps for the "Thinking" expander
    reasoning: Dict[str, any]

# --- LangGraph Nodes (The "steps" in our "flowchart") ---

def start_timer(state: AgentState):
    """Node to start the timer for metrics."""
    state["metrics"] = {"start_time": time.time()}
    state["reasoning"] = {} # Initialize reasoning dict
    return state

def detect_intent(state: AgentState):
    """
    Node 1: Detects the user's intent and identifies the most relevant policy.
    This is the "Policy Selection" feature.
    """
    question = state["question"]
    policy_info = get_policy_info()
    
    # Create a formatted string of policies for the prompt
    policy_list = "\n".join([f"- {p['number']}: {p['title']}" for p in policy_info])
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", f"""
You are an expert at routing user questions to the correct policy document.
Based on the user's question, identify the SINGLE most relevant policy from the list below.
Respond with *only* the policy number (e.g., "POL-HR-004"). Do not add any other text.
If no policy is relevant, respond with "GENERAL".

Available Policies:
{policy_list}
- GENERAL: For all other questions.
"""),
        ("user", "{question}")
    ])
    
    llm = get_llm()
    intent_chain = prompt_template | llm
    intent = intent_chain.invoke({"question": question}).content.strip()
    
    # Store the result in the state
    state["policy_intent"] = intent
    state["reasoning"]["intent"] = intent
    return state

def multi_query_retriever(state: AgentState):
    """
    Node 2: Generates multiple versions of the user's query.
    This is the "Multi-Query Retriever" feature.
    """
    question = state["question"]
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
You are an expert at generating search queries.
Based on the user's question, generate 3 additional, related search queries.
These queries should be diverse and cover different facets of the original question.
Respond with *only* the 3 queries, separated by newlines.
"""),
        ("user", "{question}")
    ])
    
    llm = get_llm()
    query_chain = prompt_template | llm
    queries_response = query_chain.invoke({"question": question}).content
    
    multi_queries = [question] + queries_response.strip().split('\n')
    
    state["multi_queries"] = multi_queries
    state["reasoning"]["multi_queries"] = multi_queries
    return state

def retrieve_docs(state: AgentState):
    """
    Node 3: Retrieves documents from the vector store using the generated queries
    and the detected policy intent.
    """
    retriever = get_retriever(get_embedding_function())
    policy_intent = state["policy_intent"]
    queries = state["multi_queries"]
    
    all_retrieved_docs = []
    
    # Create the metadata filter
    filter_dict = {}
    if policy_intent != "GENERAL":
        filter_dict = {"policy_number": policy_intent}
        
    # Invoke the retriever for each query
    for query in queries:
        # Note: LangChain retrievers now support filters directly in invoke
        # We check if the retriever supports filtering. Chroma does.
        retrieved_docs = retriever.invoke(query, config={"filter": filter_dict})
        all_retrieved_docs.extend(retrieved_docs)
    
    # De-duplicate documents based on page content
    unique_docs = {doc.page_content: doc for doc in all_retrieved_docs}.values()
    
    state["documents"] = list(unique_docs)
    state["reasoning"]["retrieved_docs"] = list(unique_docs)
    return state

def rerank_docs(state: AgentState):
    """
    Node 4: Re-ranks the retrieved documents for relevance using Cohere.
    This is the "Reranker" feature.
    """
    question = state["question"]
    documents = state["documents"]
    reranker = get_reranker()
    
    # CohereRerank expects a list of Document objects
    reranked_docs = reranker.compress_documents(
        query=question,
        documents=documents
    )
    
    state["reranked_documents"] = reranked_docs
    state["reasoning"]["reranked_docs"] = reranked_docs
    return state

def format_context_for_llm(documents: List[Document]) -> str:
    """Helper function to format docs for the final prompt."""
    formatted_context = []
    for i, doc in enumerate(documents):
        # Create a citation tag
        policy_num = doc.metadata.get('policy_number', 'N/A')
        policy_title = doc.metadata.get('policy_title', 'Unknown Policy')
        
        citation = f"[Source {i+1}: {policy_num} - {policy_title}]"
        
        # Format the context
        formatted_context.append(f"{citation}\n{doc.page_content}\n")
    
    return "\n---\n".join(formatted_context)

def generate_answer(state: AgentState):
    """
    Node 5: Generates the final, citable answer.
    This is the "Answer" and "Citation" feature.
    """
    question = state["question"]
    documents = state["reranked_documents"]
    
    if not documents:
        # Handle case where no relevant documents were found
        state["answer"] = "I'm sorry, I couldn't find any relevant policy information for your question. Please try rephrasing."
        return state

    # Format the context with citations
    context_str = format_context_for_llm(documents)
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", f"""
You are "Finance House Policy Bot," an expert assistant for Finance House employees.
Your primary goal is to provide clear, accurate, and helpful answers to questions about company policies.

**Your Instructions:**
1.  **Answer the question:** Use the "Context" provided below to answer the user's question.
2.  **Base answers on context ONLY:** Do not use any outside knowledge. If the answer is not in the context, say "I couldn't find information on that topic in the provided policies."
3.  **Cite your sources:** This is very important. For every piece of information you provide, you *must* include the source citation (e.g., `[Source 1: POL-HR-004 - Annual Leave TimeOff]`).
4.  **Be professional and neutral:** Use a clear, concise, and helpful tone.
5.  **Do not make up policies:** Stick strictly to the text.

---
**Context:**
{context_str}
---
"""),
        ("user", "{question}")
    ])
    
    llm = get_llm()
    answer_chain = prompt_template | llm
    
    # We use .stream() here to enable streaming in the UI
    answer_stream = answer_chain.stream({"question": question})
    
    # Since this node is the last one to *generate* content, 
    # we'll save the full stream to the state.
    # The UI will be responsible for rendering it.
    state["answer"] = answer_stream 
    return state

def generate_followup(state: AgentState):
    """
    Node 6: Generates 3 related follow-up questions.
    This is the "Follow-up Questions" feature.
    """
    question = state["question"]
    # We need to buffer the answer stream to get the final string
    # In a real app, we'd pass the final string. For this demo,
    # we'll just use the original question as context.
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
You are a helpful assistant. Based on the user's last question, generate exactly 3
related follow-up questions they might also want to ask.
- Keep the questions concise (10-15 words).
- Do not number them.
- Respond with *only* the 3 questions, separated by newlines.
"""),
        ("user", "{question}")
    ])
    
    llm = get_llm()
    followup_chain = prompt_template | llm
    response = followup_chain.invoke({"question": question}).content
    
    questions = [q.strip() for q in response.strip().split('\n') if q.strip()]
    
    state["follow_up_questions"] = questions
    state["reasoning"]["follow_up_questions"] = questions
    return state

def end_timer(state: AgentState):
    """Node to stop the timer and calculate total time."""
    end_time = time.time()
    total_time = end_time - state["metrics"]["start_time"]
    state["metrics"]["total_time"] = round(total_time, 2)
    state["reasoning"]["metrics"] = {"total_time": round(total_time, 2)}
    return state

# --- 3. Graph Assembly ---

@st.cache_resource
def get_graph():
    """
    Assemble the LangGraph.
    This is cached so the graph is built only once.
    """
    workflow = StateGraph(AgentState)
    
    # Add all the nodes
    workflow.add_node("start_timer", start_timer)
    workflow.add_node("detect_intent", detect_intent)
    workflow.add_node("multi_query_retriever", multi_query_retriever)
    workflow.add_node("retrieve_docs", retrieve_docs)
    workflow.add_node("rerank_docs", rerank_docs)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("generate_followup", generate_followup)
    workflow.add_node("end_timer", end_timer)
    
    # Set the entry point
    workflow.set_entry_point("start_timer")
    
    # Define the edges (the "flowchart")
    workflow.add_edge("start_timer", "detect_intent")
    workflow.add_edge("detect_intent", "multi_query_retriever")
    workflow.add_edge("multi_query_retriever", "retrieve_docs")
    workflow.add_edge("retrieve_docs", "rerank_docs")
    workflow.add_edge("rerank_docs", "generate_answer")
    workflow.add_edge("generate_answer", "generate_followup")
    workflow.add_edge("generate_followup", "end_timer")
    workflow.add_edge("end_timer", END)
    
    # Compile the graph
    return workflow.compile()

# --- 4. Streamlit UI Application ---

def inject_custom_css():
    """Injects custom CSS for a professional, neutral UI."""
    st.markdown("""
        <style>
            /* --- Base & Colors (FIX: Set default text color) --- */
            .stApp {
                background-color: #f0f2f6; /* Neutral light gray background */
                color: #1f2937; /* ADDED: Default dark text color for the whole app */
            }
            
            /* --- Chat Bubbles (FIX: Explicitly set text color) --- */
            [data-testid="chat-message-container"] {
                border-radius: 18px;
                padding-top: 10px;
                padding-bottom: 10px;
                margin-bottom: 10px;
            }
            
            /* User (You) Bubble */
            [data-testid="chat-message-container"]:has([data-testid="chat-avatar-user"]) {
                background-color: #e1f0ff; /* Light, friendly blue */
                color: #1f2937; /* ADDED: Dark text */
            }

            /* Assistant (Bot) Bubble (FIX: Explicitly set text color) */
            [data-testid="chat-message-container"]:has([data-testid="chat-avatar-assistant"]) {
                background-color: #ffffff; /* Clean white */
                border: 1px solid #d1d5db; /* Subtle border */
                box-shadow: 0 1px 3px rgba(0,0,0,0.03);
                color: #1f2937; /* ADDED: Dark text */
            }
            
            /* --- "Show Reasoning" Expander (FIX: Explicitly set text color) --- */
            [data-testid="stExpander"] {
                border: 1px solid #d1d5db;
                border-radius: 10px;
                background-color: #fafafa; /* Slightly off-white */
                margin-top: 15px;
                color: #1f2937; /* ADDED: Dark text for content inside */
            }
            [data-testid="stExpander"] summary {
                font-weight: 600;
                color: #4b5563; /* Dark gray text */
            }
            
            /* --- Follow-up Question Buttons (No change, was already OK) --- */
            .stButton > button {
                width: 100%;
                text-align: left;
                background-color: #ffffff;
                border: 1px solid #d1d5db;
                color: #1f2937;
                font-weight: 500;
                border-radius: 8px;
                transition: background-color 0.2s ease, border-color 0.2s ease;
            }
            .stButton > button:hover {
                background-color: #f9fafb; /* Very light gray hover */
                border-color: #9ca3af;
                color: #000;
            }
            .stButton > button:active {
                background-color: #f3f4f6;
            }
            
            /* --- Source Citation Styling (No change, was already OK) --- */
            .source-citation {
                font-size: 0.85rem;
                color: #6b7280; /* Medium gray */
                background-color: #f3f4f6;
                padding: 2px 6px;
                border-radius: 4px;
                display: inline-block;
                margin-right: 5px;
            }
        </style>
    """, unsafe_allow_html=True)

def format_reasoning_docs(docs: List[Document]) -> str:
    """Helper to format retrieved/reranked docs for the expander."""
    md_string = ""
    for i, doc in enumerate(docs):
        policy_num = doc.metadata.get('policy_number', 'N/A')
        policy_title = doc.metadata.get('policy_title', 'Unknown')
        md_string += f"""
<details>
    <summary><strong>Doc {i+1}: {policy_num} - {policy_title}</strong></summary>
    <p style="font-size: 0.9rem; color: #4b5563; background-color: #f9fafb; border: 1px solid #e5e7eb; padding: 10px; border-radius: 5px;">
        {doc.page_content[:500]}...
    </p>
</details>
"""
    return md_string

def run_query(app, question: str):
    """
    This function is the main entry point for processing a user's question.
    It uses st.status to show the "continuous feedback" and streams the response.
    """
    
    # --- UI: Show Continuous Feedback ---
    with st.status("Thinking...", expanded=False) as status:
        config = RunnableConfig(recursion_limit=50)
        inputs = {"question": question}
        reasoning_data = {}
        
        # This will hold the streamed answer chunks
        answer_placeholder = st.empty()
        full_answer = ""
        
        try:
            # --- Run the Graph (Agent) ---
            # We iterate through all events in the graph execution
            for event in app.stream(inputs, config=config):
                
                # event is a dict, keys are the node names
                if "start_timer" in event:
                    status.update(label="Starting analysis...")
                
                if "detect_intent" in event:
                    status.update(label="Detecting policy intent...")
                    reasoning_data["intent"] = event["detect_intent"]["policy_intent"]

                if "multi_query_retriever" in event:
                    status.update(label="Generating sub-queries...")
                    reasoning_data["multi_queries"] = event["multi_query_retriever"]["multi_queries"]

                if "retrieve_docs" in event:
                    status.update(label="Retrieving relevant documents...")
                    reasoning_data["retrieved_docs"] = event["retrieve_docs"]["documents"]

                if "rerank_docs" in event:
                    status.update(label="Reranking documents for relevance...")
                    reasoning_data["reranked_docs"] = event["rerank_docs"]["reranked_documents"]

                if "generate_answer" in event:
                    status.update(label="Generating final answer...")
                    # The 'answer' is a streaming object. We iterate it.
                    for chunk in event["generate_answer"]["answer"]:
                        full_answer += chunk.content
                        answer_placeholder.markdown(full_answer + "▌")
                
                if "generate_followup" in event:
                    status.update(label="Generating follow-up questions...")
                    reasoning_data["follow_up_questions"] = event["generate_followup"]["follow_up_questions"]
                
                if "end_timer" in event:
                    status.update(label="Done!")
                    reasoning_data["metrics"] = event["end_timer"]["metrics"]
                    status.success(f"Done in {reasoning_data['metrics']['total_time']}s!")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            print(f"Error during graph execution: {e}") # For debugging
            status.error("An error occurred.")
            return

    # --- Post-Stream Processing ---
    
    # Final cleanup of the answer
    answer_placeholder.markdown(full_answer)
    
    # Store the complete message in session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "reasoning": reasoning_data, # This is the "wow" data
        "follow_ups": reasoning_data.get("follow_up_questions", [])
    })

# --- Main Application Logic ---
def main():
    
    # --- Page Configuration ---
    st.set_page_config(
        page_title="Finance House Policy Bot",
        page_icon="🤖",
        layout="wide"
    )
    
    # Inject the professional, neutral CSS
    inject_custom_css()
    
    # --- Title and Sidebar ---
    st.title("Finance House Policy Bot 🤖")
    st.sidebar.header("About This App")
    st.sidebar.markdown("""
    This advanced chatbot is designed to help Finance House employees
    get accurate, citable answers from our 10 core company policies.
    
    **How it Works:**
    1.  **Intent Detection:** Identifies the relevant policy.
    2.  **Multi-Query:** Generates related queries to find the best info.
    3.  **Retrieve:** Fetches documents from a local vector store.
    4.  **Rerank:** Uses a Cohere model to find the *most* relevant text.
    5.  **Generate:** A Gemini LLM synthesizes the final answer with citations.
    6.  **Follow-ups:** Suggests related questions to explore.
    """)
    
    # --- Initialize Chat History ---
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello! I'm the Finance House Policy Bot. How can I help you today?",
            "reasoning": None,
            "follow_ups": []
        }]

    # --- Load the RAG Agent ---
    # This is cached, so it only loads once
    try:
        app = get_graph()
    except Exception as e:
        st.error(f"Failed to initialize the RAG agent: {e}")
        st.stop()


    # --- Display Chat History ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # --- The "Wow" Features: Reasoning & Follow-ups ---
            if msg["role"] == "assistant" and msg["reasoning"]:
                
                with st.expander("Show Reasoning 🧠"):
                    r = msg["reasoning"]
                    st.markdown(f"**Metric:** Total query time: `{r['metrics']['total_time']}s`")
                    st.markdown(f"**1. Policy Intent:** `{r['intent']}`")
                    
                    st.markdown(f"**2. Generated Sub-Queries:**")
                    st.json(r['multi_queries'])
                    
                    st.markdown(f"**3. Retrieved Documents (before reranking):**")
                    st.markdown(format_reasoning_docs(r['retrieved_docs']), unsafe_allow_html=True)
                    
                    st.markdown(f"**4. Reranked Documents (Top {RERANKER_TOP_N}):**")
                    st.markdown(format_reasoning_docs(r['reranked_docs']), unsafe_allow_html=True)

                # Display follow-up questions as clickable buttons
                if msg["follow_ups"]:
                    st.markdown("<br>", unsafe_allow_html=True)
                    cols = st.columns(len(msg["follow_ups"]))
                    for i, fup_question in enumerate(msg["follow_ups"]):
                        if cols[i].button(fup_question, use_container_width=True):
                            # When button is clicked, add to history and run query
                            st.session_state.messages.append({"role": "user", "content": fup_question})
                            with st.chat_message("user"):
                                st.markdown(fup_question)
                            
                            with st.chat_message("assistant"):
                                run_query(app, fup_question)
                            st.rerun() # Rerun to show the new message and response
            
            # Add a small divider after assistant messages
            if msg["role"] == "assistant" and msg["content"] != "Hello! I'm the Finance House Policy Bot. How can I help you today?":
                st.markdown("---", unsafe_allow_html=True)


    # --- Chat Input Box ---
    if prompt := st.chat_input("Ask a question about a company policy..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response (with streaming and "thinking" status)
        with st.chat_message("assistant"):
            run_query(app, prompt)
        
        # Rerun to clear the "follow-up" buttons from previous answers
        st.rerun()

if __name__ == "__main__":
    main()