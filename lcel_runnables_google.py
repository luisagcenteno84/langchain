from operator import itemgetter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda

def length_function(text):
    return len(text)

def _multiple_length_function(text1, text2):
    return len(text1) * len(text2)

def multiple_length_function(_dict):
    return _multiple_length_function(_dict["text1"],_dict["text2"])

prompt = ChatPromptTemplate.from_template("what is {a} + {b}. answer like a sales man")
model = ChatGoogleGenerativeAI(model="models/gemini-pro")

chain = (
    {
        "a": itemgetter("foo") | RunnableLambda(length_function),
        "b": {"text1": itemgetter("foo"), "text2": itemgetter("bar")}
        | RunnableLambda(multiple_length_function)
    }
    | prompt
    | model
)

print(chain.invoke({"foo": "primero", "bar": "cuarto"}))