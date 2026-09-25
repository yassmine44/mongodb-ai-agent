
import key_param

from typing import Annotated
from typing_extensions import TypedDict

from pymongo import MongoClient

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ============================================================
# 1. MONGODB CONNECTION
# ============================================================

def init_mongodb():
    """Initialize MongoDB Atlas and retrieve collections."""

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


# ============================================================
# 2. LOCAL EMBEDDING MODEL
# ============================================================

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
)


def generate_embedding(text: str) -> list[float]:
    """Generate a query embedding using local Ollama."""

    return embedding_model.embed_query(text)


# ============================================================
# 3. MONGODB TOOLS
# ============================================================

def create_tools(vs_collection, full_collection):
    """Create the two MongoDB documentation tools."""

    @tool
    def get_information_for_question_answering(
        user_query: str
    ) -> str:
        """
        Search MongoDB documentation using vector similarity.
        Use this tool for general technical questions about MongoDB.
        """

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
        """
        Retrieve a complete MongoDB documentation page by exact title.
        Use this tool when the user asks to summarize a specific page.
        """

        document = full_collection.find_one(
            {"title": user_query},
            {"_id": 0, "title": 1, "body": 1}
        )

        if document is None:
            return "Document not found."

        return (
            f"Title: {document['title']}\n\n"
            f"{document.get('body', '')}"
        )

    return [
        get_information_for_question_answering,
        get_page_content_for_summarization
    ]


# ============================================================
# 4. GRAPH STATE
# ============================================================

class GraphState(TypedDict):
    """Conversation messages accumulated by LangGraph."""

    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# 5. AGENT NODE
# ============================================================

def agent(
    state: GraphState,
    llm_with_tools,
    final_chain
):
    """
    Select an appropriate tool or generate the final answer.
    """

    messages = state["messages"]

    # If the previous node returned documentation,
    # produce the final answer without enabling tools again.
    if isinstance(messages[-1], ToolMessage):

        print("\nGenerating final answer from MongoDB documents...")

        question = next(
            (
                message.content
                for message in reversed(messages)
                if isinstance(message, HumanMessage)
            ),
            ""
        )

        # Collect the results of the most recent tool executions.
        observations = []

        for message in reversed(messages):
            if not isinstance(message, ToolMessage):
                break

            observations.append(message.content)

        observations.reverse()

        missing_results = {
            "No relevant documents found.",
            "Document not found."
        }

        if all(
            result.strip() in missing_results
            or result.startswith("Unknown tool:")
            for result in observations
        ):
            return {
                "messages": [
                    AIMessage(content="I DON'T KNOW")
                ]
            }

        context = "\n\n".join(observations)

        response = final_chain.invoke({
            "question": question,
            "context": context
        })

        return {"messages": [response]}

    # Otherwise, let Llama decide whether a tool is needed.
    response = llm_with_tools.invoke({
        "messages": messages
    })

    return {"messages": [response]}


# ============================================================
# 6. TOOL NODE
# ============================================================

def tool_node(state: GraphState, tools_by_name):
    """Execute tool calls requested by the language model."""

    results = []

    tool_calls = state["messages"][-1].tool_calls

    for tool_call in tool_calls:

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        print(f"\nExecuting tool: {tool_name}")
        print(f"Arguments: {tool_args}")

        if tool_name not in tools_by_name:
            observation = f"Unknown tool: {tool_name}"

        else:
            selected_tool = tools_by_name[tool_name]

            observation = selected_tool.invoke(tool_args)

        results.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"],
                name=tool_name
            )
        )

    return {"messages": results}


# ============================================================
# 7. CONDITIONAL ROUTER
# ============================================================

def route_tools(state: GraphState):
    """Route to the tool node or finish the graph."""

    messages = state["messages"]

    if not messages:
        raise ValueError("No messages found in graph state.")

    last_message = messages[-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return END


# ============================================================
# 8. BUILD THE LANGGRAPH WORKFLOW
# ============================================================

def init_graph(
    llm_with_tools,
    final_chain,
    tools_by_name
):
    """Create and compile the agent's decision-making graph."""

    graph = StateGraph(GraphState)

    graph.add_node(
        "agent",
        lambda state: agent(
            state,
            llm_with_tools,
            final_chain
        )
    )

    graph.add_node(
        "tools",
        lambda state: tool_node(
            state,
            tools_by_name
        )
    )

    graph.add_edge(START, "agent")

    graph.add_conditional_edges(
        "agent",
        route_tools,
        {
            "tools": "tools",
            END: END
        }
    )

    graph.add_edge("tools", "agent")

    return graph.compile()


# ============================================================
# 9. EXECUTE AND DISPLAY THE GRAPH
# ============================================================

def execute_graph(app, user_input: str):
    """Execute the graph and display all processing stages."""

    print("\n" + "=" * 60)
    print("USER:", user_input)
    print("=" * 60)

    input_state = {
        "messages": [
            HumanMessage(content=user_input)
        ]
    }

    final_answer = None

    for output in app.stream(
        input_state,
        config={"recursion_limit": 10},
        stream_mode="updates"
    ):

        for node_name, value in output.items():

            print(f"\nNODE: {node_name}")

            messages = value["messages"]
            last_message = messages[-1]

            if node_name == "agent":

                if getattr(last_message, "tool_calls", None):
                    print("Requested tools:")

                    for tool_call in last_message.tool_calls:
                        print("Name:", tool_call["name"])
                        print("Arguments:", tool_call["args"])

                else:
                    final_answer = last_message.content

            elif node_name == "tools":

                for message in messages:
                    print("\nTool:", message.name)
                    print("Result:")
                    print(message.content)

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    if final_answer is not None:
        print(final_answer)
    else:
        print("No final answer generated.")


# ============================================================
# 10. MAIN APPLICATION
# ============================================================

def main():
    """Initialize MongoDB, Ollama and the LangGraph agent."""

    client, vs_collection, full_collection = init_mongodb()

    try:
        print("MongoDB Atlas connected successfully!")

        # Create the MongoDB tools
        tools = create_tools(
            vs_collection,
            full_collection
        )

        tools_by_name = {
            tool.name: tool
            for tool in tools
        }

        # Initialize the local language model
        llm = ChatOllama(
            model="llama3.2",
            temperature=0
        )

        print("Ollama Llama 3.2 initialized successfully!")

        # Prompt used when deciding which tool to call
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are a technical assistant specializing in MongoDB.

                Available tools: {tool_names}

                INSTRUCTIONS:
                - Use the vector search tool for general
                  questions about MongoDB documentation.
                - Use the document retrieval tool when
                  the user requests a specific page by title.
                - Prefer using documentation over guessing.
                - Do not call tools unnecessarily.
                """
            ),
            MessagesPlaceholder(variable_name="messages")
        ])

        prompt = prompt.partial(
            tool_names=", ".join(
                tool.name for tool in tools
            )
        )

        # Bind tools to Llama 3.2
        llm_with_tools = prompt | llm.bind_tools(tools)

        # Separate prompt for the final answer.
        # Tools are deliberately disabled at this stage.
        final_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are a technical assistant specializing in MongoDB.

                Answer the user's question using ONLY the retrieved
                MongoDB documentation provided below.

                If the user requests a summary, summarize the
                relevant document clearly and concisely.

                Do not invent information that is not present
                in the retrieved documentation.

                Say I DON'T KNOW only if the provided documents
                do not contain enough information to answer.

                Do not request additional tools.
                """
            ),
            (
                "human",
                "User question:\n{question}\n\n"
                "Retrieved documentation:\n{context}\n\n"
                "Provide your final answer:"
            )
        ])

        final_chain = final_prompt | llm

        # Build the graph
        app = init_graph(
            llm_with_tools,
            final_chain,
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
