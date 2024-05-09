from operator import itemgetter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableConfig
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers.string import StrOutputParser
import json

def parse_or_fix(text: str, config: RunnableConfig):
    fixing_chain = (
        ChatPromptTemplate.from_template(
            "Fix the following text: \n\n```text\n{input}\n```Error:{error}"
            " Don't narrate, just respond with the fixed data"
        )
        | ChatGoogleGenerativeAI(model="models/gemini-pro")
        | StrOutputParser()
    )
    for _ in range(3):
        try:
            return json.loads(text)
        except Exception as e:
            text = fixing_chain.invoke({"input": text, "error": e}, config)
    return "Failed to parse"

print(RunnableLambda(parse_or_fix).invoke(
    "{foo, bar}"
))