from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# instantiate the GenAI object using Gemini Pro - Gemini Pro Vision could also be used
llm = ChatGoogleGenerativeAI(model="gemini-pro")


#creates a prompt that will be later fed into the chain
prompt = ChatPromptTemplate.from_messages([
    ("system","You are a world-class technical documentation writer"),
    ("user","{input}")
])

#instantiating a parser to generate a response String output instead of an response object
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

response = chain.invoke({"input": 'how can langsmith help with testing?'})


print(response)
