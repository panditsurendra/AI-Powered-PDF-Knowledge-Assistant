from langchain_community.retrievers import ArxivRetriever

# create the retriever
retriever = ArxivRetriever(
    load_max_docs=2,      # number of papers to retrieve
    load_all_available_meta=True
)

# query arxiv
docs = retriever.invoke("xrd")

# print results
for i, doc in enumerate(docs):
    print(f"\nResult {i+1}")
    print("Title:", doc.metadata.get("Title"))
    print("Authors:", doc.metadata.get("Authors"))
    print("Summary:", doc.page_content)  
    
    # here doc.page_content is automatically truncated to 1000 characters, you can change it by passing the parameter "max_content_length" while creating the retriever. By default it is set to 1000 characters.

# ArxivRetriever does NOT return full papers
# Abstract (summary) → stored in page_content





    