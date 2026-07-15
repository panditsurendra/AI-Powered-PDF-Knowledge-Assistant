from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv 
load_dotenv()

from langchain_mistralai import MistralAIEmbeddings

embedding_model = MistralAIEmbeddings(
    model="mistral-embed",
)

from langchain_core.documents import Document

docs = [
    Document(
        page_content="Python is a programming language used in ai and ml",
        metadata={"source": "basic.txt"}
    ),

    Document(
        page_content="Text splitting is important for RAG in Agentic ai",
        metadata={"source": "Agentic.txt"}
    ),
    Document(
        page_content="Generative models generate the text, while discriminative models classify it.",
        metadata={"source": "gen_book"}
    )
]


vector_store = Chroma.from_documents( # it willautomatcally create embeddings for the documents and store in chroma vector database
    documents=docs,
    embedding=embedding_model,
    collection_name="my_collection",
    persist_directory="./chroma_db" # it will create a folder named chroma_db in current directory and store the vector database there
)


results = vector_store.similarity_search("What is the use of python in ai?", k=2) # it will return the top 2 similar documents from the vector database based on the query  
for i in results:
    print(i.page_content)
    print(i.metadata)

# print(results[0].page_content)


retriver = vector_store.as_retriever() # it will convert the vector store into a retriever which can be used in RAG pipeline
retriver_results = retriver.invoke("What is the use of python in ai?") # it will return the relevant documents from the vector database based on the query using the retr


for i in retriver_results:
    print(i.page_content)
    print(i.metadata)