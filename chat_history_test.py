from langchain.memory import ChatMessageHistory

demo_ephemeral_chat_history = ChatMessageHistory()

demo_ephemeral_chat_history.add_user_message("hi!")
demo_ephemeral_chat_history.add_ai_message("what's up?")
print(demo_ephemeral_chat_history.messages)