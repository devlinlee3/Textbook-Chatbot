from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

FILE_PATH = "../documents/Frank Vahid - Digital Design with RTL Design, VHDL, and Verilog Solution Manual-Wiley (2010).pdf"

loader = PyPDFLoader(FILE_PATH) #load document

#split document into pages/chunks
pages = loader.load_and_split()

#print(len(pages))


#embedding functions- create embeddings of each of the chunks

embedding_function = HuggingFaceEmbeddings(
    model_name = "all-MiniLM-L6-v2"
)

#create vector store/store data into db
vectordb = Chroma.from_documents(
    documents = pages, 
    embedding = embedding_function, 
    persist_directory = "../vector_db",
    collection_name ="Frank_Vahid_textbook"
)

print(len(pages))

#make persistent/persistent data
vectordb.persist()


