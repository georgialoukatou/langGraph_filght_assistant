from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import openai
from langsmith.wrappers import wrap_openai
from langsmith import traceable
load_dotenv()

# Auto-trace LLM calls in-context
client = wrap_openai(openai.Client())

@traceable # Auto-trace this function
def pipeline(user_input: str):
    result = client.chat.completions.create(
        messages=[{"role": "user", "content": user_input}],
        model="gpt-3.5-turbo"
    )
    return result.choices[0].message.content

print(pipeline("Hello, world!"))
# Out:  Hello there! How can I assist you today?