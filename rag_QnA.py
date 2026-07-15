from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


from langchain_chroma import Chroma


llm = ChatGroq(model="openai/gpt-oss-20b")



loader = PyPDFLoader("Datas/OOPS2_cn.pdf")
docs = loader.load()

print(len(docs))

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)

splitted_data = splitter.split_documents(docs)
print(len(splitted_data))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vector_store = Chroma.from_documents(
    documents=splitted_data,
    embedding=embeddings,
    collection_name="OOPS2_cn"
)   

# query = "What is shallow copy?"

# data = vector_store.similarity_search(query) 
# print(len(data))
# print(data[0].page_content)

# context = ""
# for d in data:
#     context += d.page_content + "\n --------------------------------------- \n"    

# print(context)


# res = llm.invoke(f"""Can you provide me the answer based on
#         provided context for my quetion, context: {context} and question: {query}""")

# print(res.content)


#### Chain - Context_generate | prompt | llm | strparser
def get_context(query:str):
    data = vector_store.similarity_search(query=query)
    context = ""
    for doc in data:
        context += doc.page_content + "\n"

    return {
        "context":context,
        "question":query
    }


prompt = PromptTemplate.from_template("""
    You are a helpful assistant and provide answerd based on the context for user question. and 
    if you don't know the answer, then you can say that 'I dont know.'
    Context: {context}
    Question: {question}
""")
rag_chain = get_context | prompt | llm
res = rag_chain.invoke("What is shallow copy? ")
print(res.content)