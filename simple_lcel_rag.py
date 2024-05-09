from operator import itemgetter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.vectorstores import FAISS

vectorstore = FAISS.from_texts(
    ["Luis likes to code", "Luis likes listening to music", "Luis likes to be with his daughters", "Luis likes listening to Tool's music"],
    GoogleGenerativeAIEmbeddings(model="models/embedding-001")
)

retriever = vectorstore.as_retriever()

template = """
    Answer the questions based only on the following context:
    {context}
    Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

model = ChatGoogleGenerativeAI(model="models/gemini-pro")


chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt 
    | model
    | StrOutputParser()
)

#print(chain.invoke("what kind of things does Luis like to do?"))

template_v2 = """
    Answer the question only based on the following context:
    {context}
    Question: {question}
    Answer like a {character_type}
"""

prompt = ChatPromptTemplate.from_template(template_v2)

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "character_type": itemgetter("character_type")        
    }
    | prompt
    | model
    | StrOutputParser()
)

print(chain.invoke({"question": "what kind of music does Luis like?", "character_type": "pirate"}))

