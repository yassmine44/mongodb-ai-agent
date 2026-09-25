
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

response = llm.invoke(
    "Explain what an AI agent is in one sentence."
)

print("Réponse :", response.content)
