from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

#instantiate the Gemini Pro model from Google
llm = ChatGoogleGenerativeAI(model="gemini-pro")
#instantiate a Google GenAI embedding using the embeddings-001 model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 1. Indexing
## 1.1. Load: First we need to load our data. We’ll use DocumentLoaders for this.U sing beautifulsoup4. We load the document in this url to add them as embeddings in the LLM call
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()

## 2.2. Split: Text splitters break large Documents into smaller chunks. This is useful both for indexing data and for passing it in to a model, since large chunks are harder to search over and won’t fit in a model’s finite context window.
### create a text splitter that will be used to split the documents from the previous url
text_splitter = RecursiveCharacterTextSplitter()

### split the documents using the text splitter
documents = text_splitter.split_documents(docs)

## 1.3.Store: We need somewhere to store and index our splits, so that they can later be searched over. This is often done using a VectorStore and Embeddings model
### create a vector store (index) using the split documents as well as the embeddings from Google GenAI
vector = FAISS.from_documents(documents, embeddings)

## 2. Retrieval and Generation
## 2.1. Retrieve: Given a user input, relevant splits are retrieved from storage using a Retriever.
### create the prompt that indicates that only the context from the documents should be used to answer the question
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>                                 
                                            
<question>
{input}""")

### create a document chain leveraging the Google GenAI LLM object and the previously created prompt
document_chain = create_stuff_documents_chain(llm,prompt)

### obtain the retriever object from the vector store created previously with documents
retriever = vector.as_retriever()

### create the retrieval chain using the vector retriever and the document chain that includes the LLM and the prompt
retrieval_chain = create_retrieval_chain(retriever, document_chain)

## 2.2 Generate: A ChatModel / LLM produces an answer using a prompt that includes the question and the retrieved data
### call the invoke method with the question to be answered with the context from the documents and prompt
response = retrieval_chain.invoke({"input":"how can langchain help with testing?"})

print(response["answer"])