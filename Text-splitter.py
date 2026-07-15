
##### --- There are two types of text splitters in langchain ---------------
# RecursiveCharacterTextSplitter (most used ⭐)

from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

text = """LangChain is a framework for developing applications powered by language models. It can be used for chatbots, 
Generative Question-Answering (GQA), summarization, and much more. The core idea is that we can "chain" together different 
components to create more advanced use cases around LLMs. Chains may consist of multiple components 
from several modules, or just a single component. The most important module is the "core" module, which provides the 
building blocks for all chains."""

##### -----------RecursiveCharacterTextSplitter
# splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20)

# chunks = splitter.split_text(text)

# print(chunks)

# Semantic / Meaning-Based Splitting
# It’s an intelligent text splitter that tries to preserve meaning while splitting.

# Instead of blindly cutting text, it:

# 👉 tries multiple separators in order (hierarchy)

# Paragraphs → Lines → Words → Characters


# ["\n\n", "\n", " ", ""]

# Meaning:

# Try splitting by paragraphs
# If chunk too big → split by lines
# Still too big → split by words
# Last fallback → characters









###### -----------CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator= "",
    chunk_size = 10,
    chunk_overlap=3
)

#### Note :: Here splitting occure according to separator and 
####  Bydefault seperator is \n\n , thata if seperator is not there then it will split if there found \n\n no matter what is the chunk size and chunk overlap

# First → split text using the separator
# Then → combine pieces until chunk_size is reached
# Add overlap between chunks



# separator = "\n"
# 👉 Step 1 (split):

# ["Line1", "Line2", "Line3", "Line4"]

# 👉 Step 2 (merge into chunks):
# Depending on chunk_size, it combines them.



data = TextLoader("Datas/notes.txt")

docs = data.load()

chunks = splitter.split_documents(docs)

print(len(chunks))
print(chunks)
for i in chunks:
    print(i.page_content)
    print()
