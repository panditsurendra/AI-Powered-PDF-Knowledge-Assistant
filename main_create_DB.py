# #load pdf 
# #split into chunks 
# #create the embeddings 
# #store into chroma 
# from dotenv import load_dotenv
# load_dotenv()

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_mistralai import MistralAIEmbeddings

# embedding_model = MistralAIEmbeddings(
#     model="mistral-embed",
# )

# from langchain_community.vectorstores import Chroma 
# # from dotenv import load_dotenv

# load_dotenv()


# data = PyPDFLoader("Datas/competitive programming book.pdf")






# docs = data.load()

# # import os # if want to run this from any directory
# # base_dir = os.path.dirname(__file__)
# # file_path = os.path.join(base_dir, "..", "Datas", "competitive programming book.pdf")
# # data = PyPDFLoader(file_path)

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 1000,
#     chunk_overlap = 200
# )

# chunks = splitter.split_documents(docs)

# vectorstore = Chroma.from_documents(
#     documents= chunks,
#     embedding=embedding_model,
#     persist_directory="./vector_store/mainCreate_db"
# )













# load pdf
# split into chunks
# create embeddings
# store into chroma

from dotenv import load_dotenv
load_dotenv()

import os
import glob

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma

# =========================
# Embedding Model
# =========================

embedding_model = MistralAIEmbeddings(
    model="mistral-embed",
)

# =========================
# Load Multiple PDFs
# =========================

pdf_files = glob.glob("Datas/*.pdf")

all_docs = []

for pdf_file in pdf_files:
    
    print(f"Loading: {pdf_file}")
    
    data = PyPDFLoader(pdf_file)
    docs = data.load()

    # optional metadata
    for doc in docs:
        doc.metadata["source"] = os.path.basename(pdf_file)

    all_docs.extend(docs)

print(f"\nTotal documents loaded: {len(all_docs)}")

# =========================
# Split into Chunks
# =========================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200 
)

chunks = splitter.split_documents(all_docs)

print(f"Total chunks created: {len(chunks)}")

# =========================
# Store into Chroma
# =========================

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./vector_store/mainCreate_db"
)

vectorstore.persist()

print("Vector store created successfully!")
