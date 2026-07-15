from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

from langchain_mistralai import MistralAIEmbeddings

embedding_model = MistralAIEmbeddings(
    # model="mistral-embed", # bydefault it is set to mistral-embed, you can change it to other embedding models provided by mistral ai
)

vectorstore = Chroma( #loading an existing vector DB created in main_create_DB.py
    persist_directory="./vector_store/mainCreate_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 4,
        "fetch_k":10,
        "lambda_mult" :0.5 # it is a parameter for MMR search, it controls the diversity of the results. It ranges from 0 to 1. A higher value will return more diverse results, while a lower value will return more similar results. By default it is set to 0.5 which means it will return a balanced mix of similar and diverse results.
    }
)

llm = ChatMistralAI(model = "mistral-small-2506")

#prompt template 
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

print("-------Rag system created -----------")

print("press 0 to exit ")

while True:
    print("\n\n")
    query = input("You : ")
    if query == "0":
        break 
    
    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    
    final_prompt = prompt.invoke({
        "context" :context,
        "question": query
    })
    
    response = llm.invoke(final_prompt)

    print(f"\n AI: {response.content}")


# docs = retriever.invoke(query)
# Internally it does 3 steps automatically:

# Step 1: Convert query → embedding
# embedding_model.embed_query(query)

# Step 2: Search in vector DB
# search_type="mmr"

# Step 3: Return documents
# docs = [Document, Document, Document, Document]



    