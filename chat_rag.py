from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_vertexai import VertexAIEmbeddings
from google.cloud import bigquery
from langchain.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import BigQueryVectorSearch

from langchain.chains.combine_documents import create_stuff_documents_chain

from typing import Dict
from langchain_core.runnables import RunnablePassthrough

chat = ChatGoogleGenerativeAI(model='models/gemini-pro-1.5-pro-latest', temperature=0.2)

#print(chat)



'''prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability"
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

chain = prompt | chat
'''

'''print(chain.invoke(
    {
        "messages": [
            HumanMessage(content="What should I know about coffee from nespresso brand?"),
            AIMessage(content="You should know that this coffee is really tasty and if you buy too much, it can get expensive"),
            HumanMessage(content="what did you just say?")
        ]
    }
))'''

'''demo_ephemeral_chat_history = ChatMessageHistory()

demo_ephemeral_chat_history.add_user_message(HumanMessage(content="What should I know about coffee from nespresso brand?"))
demo_ephemeral_chat_history.add_ai_message(chain.invoke({"messages":demo_ephemeral_chat_history.messages}))
demo_ephemeral_chat_history.add_user_message(HumanMessage(content="what did you just say?"))
print(chain.invoke({"messages": demo_ephemeral_chat_history.messages}))
'''        

'''loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
'''
PROJECT_ID = "solid-sun-418711"
REGION = "US"
DATASET = "chat_rag"
TABLE = "docs_and_vectors"

client = bigquery.Client(project=PROJECT_ID, location=REGION)
client.create_dataset(dataset=DATASET, exists_ok=True)

embedding = VertexAIEmbeddings(
    model_name="textembedding-gecko@latest", project_id=PROJECT_ID
)

vectorstore = BigQueryVectorSearch(
    project_id=PROJECT_ID,
    dataset_name=DATASET,
    table_name=TABLE,
    location=REGION,
    embedding=embedding,
    distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
)

#vectorstore.add_documents(all_splits)

retriever = vectorstore.as_retriever(k=4)

docs = retriever.invoke("what are socrates' sources?")

#print(docs)

question_answering_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer the user's questions based only on the below context. if you don't find the answer in this context, reply with an apology that you are not able to find the answer: \n\n{context}"
    ),
    MessagesPlaceholder(variable_name="messages")
])

document_chain = create_stuff_documents_chain(chat,question_answering_prompt)

demo_ephemeral_chat_history = ChatMessageHistory()
demo_ephemeral_chat_history.add_user_message("what are socrates sources?")


def parse_retriever_input(params: Dict):
    return params["messages"][-1].content

retrieval_chain = RunnablePassthrough.assign(
    context = parse_retriever_input | retriever,
).assign(
    answer=document_chain
)

response = retrieval_chain.invoke(
    {
        "messages": demo_ephemeral_chat_history.messages
    }
)
#print(response)
print(response["answer"])

'''demo_ephemeral_chat_history.add_ai_message(response['answer'])

demo_ephemeral_chat_history.add_user_message("tell me more about that!")

response2 = retrieval_chain.invoke(
    {
        "messages": demo_ephemeral_chat_history.messages
    }
)

#print(response2)

retrieval_chain_with_only_answer = RunnablePassthrough.assign(
    context = parse_retriever_input | retriever,
) | document_chain

answer = retrieval_chain_with_only_answer.invoke(
    {
        "messages": demo_ephemeral_chat_history.messages
    }
)

print(answer)'''