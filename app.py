import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_core.runnables.history import RunnableWithMessageHistory

from utils.utils import MODEL_MAPPING, PDF_PATH, CHROMA_DB_PATH
from utils.prompts import System_Prompt, Contextualize_q_system_prompt
 
# Load Environment Variables
load_dotenv()

# API Keys
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Questor - Study Assistant", layout="centered")


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def load_or_create_chroma_db(isCreate=False):
    """Loads existing Chroma DB or creates a new one from all PDFs in PDF_PATH folder."""
    all_docs = []
    
    for file_name in os.listdir(PDF_PATH):
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PDF_PATH, file_name))
            all_docs.extend(loader.load())
    
    if not os.path.exists(CHROMA_DB_PATH) or isCreate:
        # st.info("Rebulinding the knowledge base...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        final_documents = text_splitter.split_documents(all_docs)

        vectordb = Chroma.from_documents(
            documents=final_documents,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH
        )
        return vectordb
    else:
        return Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)


with st.sidebar:
    st.title("Questor - A Study Assistant")
    st.write("**Chatbot Settings**")
    selected_model = st.selectbox("Select Model", list(MODEL_MAPPING.keys()))
    model_name = MODEL_MAPPING[selected_model]

    uploaded_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            file_path = os.path.join(PDF_PATH, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
        
        # st.success("Documents uploaded successfully. Rebuilding knowledge base...")
        vectordb = load_or_create_chroma_db(isCreate=True)
        # st.success("Knowledge base updated!")

llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)


qa_prompt = ChatPromptTemplate.from_messages([
    ("system", System_Prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", Contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


vectordb = load_or_create_chroma_db()
retriever = vectordb.as_retriever()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm Questor, your intelligent study companion. How can I assist you in your exam preparation today?"}
    ]

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
rag_agent = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: st.session_state.chat_history,
    input_messages_key="input",
    history_messages_key='chat_history',
    output_messages_key="answer",
)

# Display chat history
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"], unsafe_allow_html=True)
        if message["role"] == "assistant" and "caption" in message:
            st.caption(message["caption"])

# User input
if user_prompt := st.chat_input("Ask me anything about your studies or exams..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_prompt)
    
    try:
        start = time.process_time()
        response = rag_agent.invoke({"input": user_prompt}, config={"configurable": {"session_id": "chat_session"}})
        elapsed_time = time.process_time() - start
        bot_response = response["answer"]

        st.session_state.messages.append({
            "role": "assistant", 
            "content": bot_response,
            "caption": f"🟡 *Response generated by {selected_model}* – ⏳ {elapsed_time:.2f} sec"
        })
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(bot_response, unsafe_allow_html=True)
            st.caption(f"🟡 *Response generated by {selected_model}* – ⏳ {elapsed_time:.2f} sec")

    except Exception as e:
        st.error(f"⚠️ Error generating response: {str(e)}")
