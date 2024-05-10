from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers.string import StrOutputParser

model = ChatGoogleGenerativeAI(model="models/gemini-pro")

prompt = ChatPromptTemplate.from_template("You are an expert in metal rock and you know a lot about bands such as Tool and A Perfect Circle and thier members and history. You will not answer any questions that are not related to these bands and their members. Question: {question}; Answer: ")


chain = prompt | model | StrOutputParser()

#print(chain.invoke({"who is the lead singer of tool and where does he live?"}))
#print(chain.invoke({"question": "what is the meaning of life?"}))
#print(chain.invoke({"question": "what is the political stance of tool?"}))
print(chain.invoke("who is the guitar player in A Perfect Circle and where does he live?"))