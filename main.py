
import key_param
from pymongo import MongoClient
from langchain_ollama import ChatOllama


def init_mongodb():
    """Initialize MongoDB client and collections."""

    mongodb_client = MongoClient(
        key_param.mongodb_uri,
        serverSelectionTimeoutMS=5000
    )

    # Verify the connection
    mongodb_client.admin.command("ping")

    DB_NAME = "ai_agents"

    vs_collection = mongodb_client[DB_NAME]["chunked_docs"]
    full_collection = mongodb_client[DB_NAME]["full_docs"]

    return mongodb_client, vs_collection, full_collection


def main():
    """Initialize MongoDB and the local AI model."""

    mongodb_client, vs_collection, full_collection = init_mongodb()

    try:
        print("MongoDB Atlas connected successfully!")

        llm = ChatOllama(
            model="llama3.2",
            temperature=0
        )

        response = llm.invoke(
            "Explain the role of MongoDB in an AI agent."
        )

        print("\nAI response:")
        print(response.content)

        print("\nCollections:")
        print(vs_collection.name)
        print(full_collection.name)

    finally:
        mongodb_client.close()


if __name__ == "__main__":
    main()
