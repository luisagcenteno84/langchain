from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI()

prompt = ChatPromptTemplate.from_messages([
    ("system","You are a world-class grill master that advises beginners while shopping"),
    ("user","{input}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

response = chain.invoke({"input": 'what is the best charcoal grill in terms of cost-benefit ratio?'})


print(response["answer"])
