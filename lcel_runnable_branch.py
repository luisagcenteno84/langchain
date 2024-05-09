from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain.schema.runnable import RunnableBranch

decision_chain = (
    PromptTemplate.from_template(
        """
            Given the user question below, classify it as either being about `Formula 1` or `Programming Languages`, or `Other`.

            Do not respond with more than one word.

            <question>
            {question}
            </question>

            Classification:
        """
    )
    | ChatGoogleGenerativeAI(model="models/gemini-pro")
    | StrOutputParser()
)

#print(chain.invoke({"question": "who is the best driver currently?"}))
#Formula 1
#print(chain.invoke({"question": "how do I create a list in python?"}))
#Programming Languages


formula1_chain = (PromptTemplate.from_template(
    """
        You are an expert in Formula 1. \
        Always answer questions starting with "In the exciting world of F1". \
        Respond to the following question:

        Question: {question}
        Answer:
    """
    )
    | ChatGoogleGenerativeAI(model="models/gemini-pro")
)

programming_chain = (PromptTemplate.from_template(
    """You are an expert in programming languages. \
        Always answer questions starting with "As an expert in Programming Languages".\
        Respond to the following question:

        Question: {question}
        Answer:
    """
    )
    | ChatGoogleGenerativeAI(model="models/gemini-pro")
)

general_chain = (
    PromptTemplate.from_template(
        """Respond to the following question:
        
        Question: {question}
        """
    )
    |ChatGoogleGenerativeAI(model="models/gemini-pro")
)

branch = RunnableBranch(
    (lambda x: "formula 1" in x["topic"].lower(), formula1_chain),
    (lambda x: "programming languages" in x["topic"].lower(), programming_chain),
    general_chain
)

full_chain = {"topic": decision_chain, "question": lambda x: x["question"]} | branch

#print(full_chain.invoke({"question": "who is the best driver right now?"}))

#print(full_chain.invoke({"question": "how do I create a list in python?"}))

print(full_chain.invoke({"question": "who has won in Italy the most?"}))