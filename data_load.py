# from dotenv import load_dotenv

# load_dotenv()

from langchain_community.document_loaders import TextLoader, PyPDFLoader

# # text loader
# TXTloader = TextLoader("Datas/text_data.txt")

# TXTdocuments = TXTloader.load()
# print(len(TXTdocuments))
# print(type(TXTdocuments))
# print(TXTdocuments)


PDFLoader = PyPDFLoader("Datas/OOPS2_cn.pdf")

PDFdocuments = PDFLoader.load()
print(len(PDFdocuments))
print(type(PDFdocuments))
# print(PDFdocuments)

# for doc in PDFdocuments:
#     print(f"\n{doc}")


print(PDFdocuments[0])
print("\n")
print(PDFdocuments[1])
