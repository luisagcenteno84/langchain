from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory

chat = ChatGoogleGenerativeAI(model='models/gemini-1.5-pro-latest', temperature=0.2)

#print(chat)



prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability"
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

chain = prompt | chat


'''print(chain.invoke(
    {
        "messages": [
            HumanMessage(content="What should I know about coffee from nespresso brand?"),
            AIMessage(content="You should know that this coffee is really tasty and if you buy too much, it can get expensive"),
            HumanMessage(content="what did you just say?")
        ]
    }
))'''

demo_ephemeral_chat_history = ChatMessageHistory()

demo_ephemeral_chat_history.add_user_message(HumanMessage(content="What should I know about coffee from nespresso brand?"))
demo_ephemeral_chat_history.add_ai_message(chain.invoke({"messages":demo_ephemeral_chat_history.messages}))
demo_ephemeral_chat_history.add_user_message(HumanMessage(content="what did you just say?"))
print(chain.invoke({"messages": demo_ephemeral_chat_history.messages}))
                            

