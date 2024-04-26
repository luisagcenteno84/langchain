import getpass
import os
from langchain_google_vertexai import ChatVertexAI
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


#os.environ['GOOGLE_API_KEY'] = getpass.getpass()

llm = ChatVertexAI(model="gemini-pro")

#bs4_strainer = bs4.SoupStrainer(class_=("post-content","post-title","post-header"))


loader = WebBaseLoader("https://en.wikipedia.org/wiki/Plato")

docs = loader.load()

#print(docs)
#print(len(docs))
#print(len(docs[0].page_content))
#print(docs[0].page_content[0:500])



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100
)

splits = text_splitter.split_documents(docs)

#print(splits)
#print(len(splits))

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

#print(vectorstore.index)
#print(vectorstore.index.ntotal)
#print(vectorstore.similarity_search("what was the name of the academy that plato founded?"))
#print(vectorstore.similarity_search_with_score("what was the name of the academy that plato founded?"))
#vector = embeddings.embed_query("what was the name of the academy that plato founded?")
#print(vectorstore.similarity_search_by_vector(vector))

retriever = vectorstore.as_retriever()

#print(retriever.invoke("what was the name of the academy that plato founded?"))


prompt = hub.pull("rlm/rag-prompt")

#print(prompt)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


#print(rag_chain.invoke("where did plato live?"))

print(rag_chain.invoke("who are the musicians in the band tool?"))
