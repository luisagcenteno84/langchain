from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("tell me a joke about {foo}")
model = ChatGoogleGenerativeAI(model="models/gemini-pro")
chain = prompt | model

print(chain.invoke({"foo":"bears"}))