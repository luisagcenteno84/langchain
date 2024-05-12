from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers.string import StrOutputParser

model = GoogleGenerativeAI(model="models/gemini-pro")
#question = "what is the meaning of life?"
#question = "how to take care of my chocolate labrador?"

question = "how to take care of my lawn?"

classify_chain = (
    PromptTemplate.from_template("Classify this question as `Philosopy` or `Pets` or `Other`. Do not explain why, just reply with one word. question:{question}. classification:")
    | model 
    | StrOutputParser()
)





#print(prompt.invoke({"question": question}))

#messages=[HumanMessage(content='')]



#chain0 = prompt | model

#print(chain0.invoke({"question": question}))
#content='',response_metadata={}, 'finish_reason':'', 'safety_ratings': [{}] id=''

#chain1 = prompt | model | StrOutputParser()
#print(chain1.invoke({"question": question}))


chain_philo = (PromptTemplate.from_template("You are an expert of classic greek philosophy. You will always respond in 100 words or less. You will always start your answer with `Philosophers overtime have reasoned that, `. You will always provide the philosopher name and the timeframe where they live. You will not answer any questions that are not related to classical greek philosophy. question: {question}; answer:")
    | model 
    | StrOutputParser()
)
chain_pet = (PromptTemplate.from_template("You are an expert taking care of pets. You will always respond in 100 words or less. You will always respond like a dog. You will not answer any questions that are not related to classical greek philosophy. question: {question}; answer:")
    | model 
    | StrOutputParser()
)

chain_generic = (PromptTemplate.from_template("Respond to the following question. Question:{question}; Answer:")
    | model
    | StrOutputParser()
)

branch = RunnableBranch(
    (lambda x: 'philosophy' in x["topic"].lower(), chain_philo),
    (lambda x: 'pets' in x["topic"].lower(),chain_pet),
    chain_generic
)

full_chain = {"topic":classify_chain, "question": lambda x: x["question"]} | branch

print(full_chain.invoke({"question": question}))