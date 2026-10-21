import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"))

response = client.chat.completions.create(
    model=os.getenv("LLM_MODEL"),
    messages=[
        {"role": "user", "content": "Summarize the purpose of the NIST AI RMF in 3 sentences."}
    ]
)

print(response.choices[0].message.content)
print(response.usage)