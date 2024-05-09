from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate


from langchain.schema.runnable import RunnableParallel, RunnablePassthrough



runnable = RunnableParallel(
    passed=RunnablePassthrough(),
    extra=RunnablePassthrough.assign(mult=lambda x: x["num"]*3),
    modified=lambda x: x["num"]+1
)

print(runnable.invoke({"num":1}))

#Output:
#{'passed': {'num': 1}, 'extra': {'num': 1, 'mult': 3}, 'modified': 2}

#Explanation:
#all three Runnables ran in parallel
#passed: RunnablePassthrough passes the value it received onto the next step
#extra: RunnablePassthrough.assign() adds the calculated value to the dictionary, along with the input
#modified: overwrites the value with the output of the lambda function

