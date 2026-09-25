
import key_param

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

from pymongo import MongoClient
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings


# 1. Connect to MongoDB Atlas
def init_mongodb():
    client = MongoClient(
        key_param.mongodb_uri,
        serverSelectionTimeoutMS=5000
    )

    client.admin.command("ping")

    db = client["ai_agents"]

    return (
        client,
        db["chunked_docs"],
        db["full_docs"]
    )


# 2. Initialize local embedding model
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
)


def generate_embedding(text: str) -> list[float]:
    """Generate an embedding using Ollama."""

    return embedding_model.embed_query(text)


# 3. Create our MongoDB tools
def create_tools(vs_collection, full_collection):

    @tool
    def get_information_for_question_answering(
        user_query: str
    ) -> str:
        """Search MongoDB documentation to answer a question."""

        query_embedding = generate_embedding(user_query)

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": 5
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "title": 1,
                    "body": 1,
                    "score": {
                        "$meta": "vectorSearchScore"
                    }
                }
            }
        ]

        results = list(vs_collection.aggregate(pipeline))

        if not results:
            return "No relevant documents found."

        context = "\n\n".join(
            f"Title: {doc.get('title', '')}\n"
            f"Score: {doc.get('score', 0):.3f}\n"
            f"Content: {doc.get('body', '')}"
            for doc in results
        )

        return context

    @tool
    def get_page_content_for_summarization(
        user_query: str
    ) -> str:
        """Retrieve a complete MongoDB document by its exact title."""

        document = full_collection.find_one(
            {"title": user_query},
            {"_id": 0, "body": 1}
        )

        if document:
            return document["body"]

        return "Document not found."

    return [
        get_information_for_question_answering,
        get_page_content_for_summarization
    ]



def main():
    """Initialize the LLM and give it access to MongoDB tools."""

    client, vs_collection, full_collection = init_mongodb()

    try:
        print("MongoDB Atlas connected successfully!")

        # Initialize our two MongoDB tools
        tools = create_tools(
            vs_collection,
            full_collection
        )

        # Initialize local Ollama model
        llm = ChatOllama(
            model="llama3.2",
            temperature=0
        )

        # Create the agent prompt
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are a helpful AI assistant specializing in MongoDB.

                You have access to tools for searching and retrieving
                technical documentation stored in MongoDB.

                Use get_information_for_question_answering for
                general questions that require documentation.

                Use get_page_content_for_summarization when the user
                requests a specific documentation page by title.

                Use the appropriate tool rather than guessing.
                Do not call tools unnecessarily.
                If there is insufficient information, say I DON'T KNOW.

                Available tools: {tool_names}
                """
            ),
            MessagesPlaceholder(variable_name="messages")
        ])

        prompt = prompt.partial(
            tool_names=", ".join(tool.name for tool in tools)
        )

        # Give Llama 3.2 access to both tools
        llm_with_tools = prompt | llm.bind_tools(tools)

        # Test automatic tool selection
        questions = [
            "What are some best practices for data backups in MongoDB?",
            "Retrieve the page titled Create a MongoDB Deployment."
        ]

        for question in questions:
            print("\nUSER:", question)

            response = llm_with_tools.invoke({
                "messages": [
                    HumanMessage(content=question)
                ]
            })

            print("TOOL CALLS:", response.tool_calls)

            if not response.tool_calls:
                print("MODEL RESPONSE:", response.content)

    finally:
        client.close()


if __name__ == "__main__":
    main()


