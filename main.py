
import key_param

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

from pymongo import MongoClient
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings


from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_ollama import ChatOllama

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

# Define the graph state
class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Agent node
def agent(state: GraphState, llm_with_tools):
    """Run the LLM using the current conversation."""

    response = llm_with_tools.invoke({
        "messages": state["messages"]
    })

    return {"messages": [response]}


# Tool execution node
def tool_node(state: GraphState, tools_by_name):
    """Execute the tools requested by the LLM."""

    results = []

    tool_calls = state["messages"][-1].tool_calls

    for tool_call in tool_calls:
        tool_name = tool_call["name"]

        print(f"\nExecuting tool: {tool_name}")

        if tool_name not in tools_by_name:
            observation = f"Unknown tool: {tool_name}"
        else:
            selected_tool = tools_by_name[tool_name]

            observation = selected_tool.invoke(
                tool_call["args"]
            )

        results.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"],
                name=tool_name,
            )
        )

    return {"messages": results}


# Conditional routing
def route_tools(state: GraphState):
    """Determine whether to execute tools or finish."""

    messages = state["messages"]

    if not messages:
        raise ValueError("No messages in graph state.")

    last_message = messages[-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return END


# Build the LangGraph workflow
def init_graph(llm_with_tools, tools_by_name):

    graph = StateGraph(GraphState)

    graph.add_node(
        "agent",
        lambda state: agent(state, llm_with_tools)
    )

    graph.add_node(
        "tools",
        lambda state: tool_node(state, tools_by_name)
    )

    graph.add_edge(START, "agent")

    graph.add_conditional_edges(
        "agent",
        route_tools,
        {
            "tools": "tools",
            END: END,
        }
    )

    # Send tool results back to the agent
    graph.add_edge("tools", "agent")

    return graph.compile()


def execute_graph(app, user_input: str):
    """Execute the graph and display its output."""

    print("\n" + "=" * 60)
    print("USER:", user_input)
    print("=" * 60)

    input_state = {
        "messages": [HumanMessage(content=user_input)]
    }

    final_answer = None

    for output in app.stream(
        input_state,
        config={"recursion_limit": 10},
        stream_mode="updates"
    ):
        for node_name, value in output.items():

            print(f"\nNODE: {node_name}")

            last_message = value["messages"][-1]

            if node_name == "agent":
                if last_message.tool_calls:
                    print(
                        "Requested tools:",
                        last_message.tool_calls
                    )
                else:
                    final_answer = last_message.content

            elif node_name == "tools":
                print("Tool result:")
                print(str(last_message.content)[:1000])

    print("\n--- FINAL ANSWER ---")

    if final_answer is not None:
        print(final_answer)
    else:
        print("No final answer generated.")



def main():
    """Initialize and run the MongoDB AI agent."""

    client, vs_collection, full_collection = init_mongodb()

    try:
        print("MongoDB Atlas connected successfully!")

        # Initialize the MongoDB tools
        tools = create_tools(
            vs_collection,
            full_collection
        )

        tools_by_name = {
            tool.name: tool
            for tool in tools
        }

        # Initialize Llama 3.2 locally
        llm = ChatOllama(
            model="llama3.2",
            temperature=0
        )

        # Agent instructions
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are an AI assistant specializing in MongoDB.

                Use the available tools to retrieve documentation.

                For general questions, use the vector search tool.

                For specific documentation pages, use the
                document retrieval tool.

                Base your answers on retrieved documents.
                Do not repeat tool calls unnecessarily.

                If the retrieved information is insufficient,
                say I DON'T KNOW.

                Available tools: {tool_names}
                """
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])

        prompt = prompt.partial(
            tool_names=", ".join(
                tool.name for tool in tools
            )
        )

        # Give Llama access to the tools
        llm_with_tools = prompt | llm.bind_tools(tools)

        # Build the graph
        app = init_graph(
            llm_with_tools,
            tools_by_name
        )

        # Test 1: Vector search
        execute_graph(
            app,
            "What are some best practices for "
            "data backups in MongoDB?"
        )

        # Test 2: Document retrieval
        execute_graph(
            app,
            "Give me a summary of the page titled "
            "Create a MongoDB Deployment"
        )

    finally:
        client.close()



if __name__ == "__main__":
    main()


