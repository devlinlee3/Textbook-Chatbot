import os

from dotenv import load_dotenv

import streamlit as st
from groq import Groq
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))#get absolute path of file

def load_document(file_path): #read and extract text from file
    loader = UnstructuredPDFLoader(file_path)
    documents = loader.load()
    return documents
# streamlit page configuration

def setup_vectorstore(documents): #convert text into vector embeddings
    embeddings = HuggingFaceEmbeddings() #embedding 

    text_splitter = CharacterTextSplitter( #split into smaller text
        separator="/n",
        chunk_size=1000,
        chunk_overlap=200
    )

    doc_chunks = text_splitter.split_documents(documents) 
    vectorstore = FAISS.from_documents(doc_chunks, embeddings) #store into faiss vectordb
    return vectorstore

def create_chain(vectorstore):
    llm = ChatGroq(
        model = "llama-3.1-70b-versatile",
        temperature=0 #randomness
    )
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )
    return chain

st.set_page_config( #set up page with streamlit
    page_title="Texbook Chatbot",
    page_icon="📕",
    layout="centered"
)

st.title("Chat with your Textbook") #title of tab

# initialize the chat history in streamlit session state

if "chat_history" not in st.session_state: #store past questions
    st.session_state.chat_history = []


uploaded_file = st.file_uploader(
    label="Upload your pdf file",
    type=["pdf"]
)

if uploaded_file: #store file if uploaded
    file_path = f"{working_dir}/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = setup_vectorstore(load_document(file_path))

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)
    
for message in st.session_state.chat_history:#display previous q&a
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask textbook")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"): #user msg
        st.markdown(user_input)
    
    with st.chat_message("assistant"): #ai msg
        response = st.session_state.conversation_chain({"question": user_input})
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "user", "content": assistant_response})