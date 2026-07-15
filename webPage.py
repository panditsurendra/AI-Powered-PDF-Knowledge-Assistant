from langchain_community.document_loaders import WebBaseLoader
# for web site Loader, pip install beautifulsoup4


url = "https://www.apple.com/in/macbook-pro/"

data = WebBaseLoader(url)

docs = data.load()

print(len(docs))
print(docs[0].page_content)





























