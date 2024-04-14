from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Create the LLM model

llm = ChatGoogleGenerativeAI(model='gemini-pro')


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

loader = WebBaseLoader('https://docs.smith.langchain.com/user_guide')

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter()

documents = text_splitter.split_documents(docs)

vector = FAISS.from_documents(documents, embeddings)

retriever = vector.as_retriever()


prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user","{input}. \nGiven the above conversation, generate a search query to look up to get information relevant to this conversation")
])

retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

chat_history = [HumanMessage(content="Can LangSmith help test my LLM Applications?"),AIMessage(content="Yes!")]

retriever_chain.invoke({
     "chat_history": chat_history,
    "input":"Tell me how"
    })




prompt = ChatPromptTemplate.from_messages([
    ("system","Answer the user's questions based on the below context: \n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user","{input}"),
])

document_chain = create_stuff_documents_chain(llm,prompt)

retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

chat_history = [HumanMessage(content="Can Langchain help test my LLM applications?"),AIMessage(content="Yes!")]

response = retrieval_chain.invoke({
        "chat_history":chat_history,
        "input":"Tell me how"
    })

print(response["answer"])