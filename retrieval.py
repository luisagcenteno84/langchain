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

#using beautifulsoup4, we load the document in this url to add them as embeddings in the LLM call
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()

#instantiate a Google GenAI embedding using the embeddings-001 model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#create a text splitter that will be used to split the documents from the previous url
text_splitter = RecursiveCharacterTextSplitter()

#split the documents using the text splitter
documents = text_splitter.split_documents(docs)

#create a vector store (index) using the split documents as well as the embeddings from Google GenAI
vector = FAISS.from_documents(documents, embeddings)

#create the prompt that indicates that only the context from the documents should be used to answer the question
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>                                 
                                            
<question>
{input}""")

#create a document chain leveraging the Google GenAI LLM object and the previously created prompt
document_chain = create_stuff_documents_chain(llm,prompt)

#obtain the retriever object from the vector store created previously with documents
retriever = vector.as_retriever()

#create the retrieval chain using the vector retriever and the document chain that includes the LLM and the prompt
retrieval_chain = create_retrieval_chain(retriever, document_chain)

#call the invoke method with the question to be answered with the context from the documents and prompt
response = retrieval_chain.invoke({"input":"how can langchain help with testing?"})

print(response["answer"])