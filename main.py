
import key_param

from typing import Annotated
from typing_extensions import TypedDict

from pymongo import MongoClient

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.tools import tool
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.mongodb import MongoDBSaver


# ============================================================
# 1. MONGODB CONNECTION
# ============================================================

DB_NAME = "ai_agents"
VECTOR_INDEX = "vector_index"


def init_mongodb():
    """Connect to MongoDB Atlas."""

    client = MongoClient(
        key_param.mongodb_uri,
        serverSelectionTimeoutMS=5000
    )

    client.admin.command("ping")

    db = client[DB_NAME]

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
    """Generate a query embedding using Ollama."""

    return embedding_model.embed_query(text)


# ============================================================
# 3. MONGODB TOOLS
# ============================================================

def create_tools(vs_collection, full_collection):
    """Create the two MongoDB tools."""

    @tool
    def get_information_for_question_answering(
        user_query: str
    ) -> str:
        """
        Search MongoDB documentation using vector similarity.
        Use for general questions about MongoDB.
        """

        query_embedding = generate_embedding(user_query)

        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": 5,
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "title": 1,
                    "body": 1,
                    "score": {
                        "$meta": "vectorSearchScore"
                    },
                }
            },
        ]

        results = list(vs_collection.aggregate(pipeline))

        if not results:
            return "No relevant documents found."

        return "\n\n".join(
            f"Title: {doc.get('title', '')}\n"
            f"Score: {doc.get('score', 0):.3f}\n"
            f"Content: {doc.get('body', '')}"
            for doc in results
        )

    @tool
    def get_page_content_for_summarization(
        user_query: str
    ) -> str:
        """
        Retrieve a complete MongoDB documentation page
        by its exact title for summarization.
        """

        document = full_collection.find_one(
            {"title": user_query},
            {
                "_id": 0,
                "title": 1,
                "body": 1,
            }
        )

        if document is None:
            return "Document not found."

        return (
            f"Title: {document['title']}\n\n"
            f"{document.get('body', '')}"
        )

    return [
        get_information_for_question_answering,
        get_page_content_for_summarization,
    ]


# ============================================================
# 4. GRAPH STATE
# ============================================================

class GraphState(TypedDict):
    """
    Store the messages belonging to the conversation.
    """

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


# ============================================================
# 5. AGENT NODE
# ============================================================

def agent(state: GraphState, llm_with_tools, final_chain):
    """Select tools, answer from documents or recall conversation."""

    messages = state["messages"]
    last_message = messages[-1]

    # 1. Handle the memory exercise without calling MongoDB tools.
    if isinstance(last_message, HumanMessage):

        question = last_message.content.strip().lower()
        question = question.rstrip("?.! ")

        if question in {
            "what did i just ask you",
            "what was my previous question",
            "quelle était ma question précédente",
            "qu'est-ce que je viens de te demander"
        }:
            previous_question = next(
                (
                    message.content
                    for message in reversed(messages[:-1])
                    if isinstance(message, HumanMessage)
                ),
                None
            )

            if previous_question is None:
                answer = "This is your first question."
            else:
                answer = f"You previously asked: {previous_question}"

            return {
                "messages": [
                    AIMessage(content=answer)
                ]
            }

    # 2. Generate the final answer after executing an external tool.
    if isinstance(last_message, ToolMessage):

        print(
            "\nGenerating final answer "
            "from MongoDB documents..."
        )

        question = next(
            (
                message.content
                for message in reversed(messages)
                if isinstance(message, HumanMessage)
            ),
            ""
        )

        observations = []

        for message in reversed(messages):
            if not isinstance(message, ToolMessage):
                break

            observations.append(message.content)

        observations.reverse()

        valid_observations = [
            result
            for result in observations
            if result.strip() not in {
                "No relevant documents found.",
                "Document not found."
            }
            and not result.startswith("Unknown tool:")
        ]

        if not valid_observations:
            return {
                "messages": [
                    AIMessage(content="I DON'T KNOW")
                ]
            }

        context = "\n\n".join(valid_observations)

        response = final_chain.invoke({
            "question": question,
            "context": context
        })

        return {"messages": [response]}

    # 3. For other questions, let Llama 3.2 select a tool.
    response = llm_with_tools.invoke({
        "messages": messages
    })

    return {"messages": [response]}

# ============================================================
# 6. TOOL EXECUTION NODE
# ============================================================

def tool_node(
    state: GraphState,
    tools_by_name
):
    """Execute tools requested by Llama 3.2."""

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

            observation = selected_tool.invoke(
                tool_args
            )

        results.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"],
                name=tool_name,
            )
        )

    return {"messages": results}


# ============================================================
# 7. CONDITIONAL ROUTER
# ============================================================

def route_tools(state: GraphState):
    """
    Route to tool execution if necessary.
    Otherwise finish the graph.
    """

    messages = state["messages"]

    if not messages:
        raise ValueError(
            "No messages found in graph state."
        )

    last_message = messages[-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return END


# ============================================================
# 8. BUILD GRAPH WITH MONGODB MEMORY
# ============================================================

def init_graph(
    llm_with_tools,
    final_chain,
    tools_by_name,
    mongodb_client
):
    """
    Compile the LangGraph agent with persistent
    MongoDB conversation memory.
    """

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
            END: END,
        }
    )

    graph.add_edge("tools", "agent")

    # Store conversation states in MongoDB Atlas.
    checkpointer = MongoDBSaver(
        mongodb_client,
        db_name=DB_NAME,
    )

    return graph.compile(
        checkpointer=checkpointer
    )


# ============================================================
# 9. EXECUTE GRAPH WITH CONVERSATION MEMORY
# ============================================================

def execute_graph(
    app,
    thread_id: str,
    user_input: str
):
    """Execute a message within a persistent conversation."""

    print("\n" + "=" * 60)
    print("THREAD:", thread_id)
    print("USER:", user_input)
    print("=" * 60)

    input_state = {
        "messages": [
            HumanMessage(content=user_input)
        ]
    }

    config = {
        "configurable": {
            "thread_id": thread_id
        },
        "recursion_limit": 10,
    }

    # Execute and display each graph node.
    for output in app.stream(
        input_state,
        config=config,
        stream_mode="updates"
    ):

        for node_name, value in output.items():

            print(f"\nNODE: {node_name}")

            messages = value["messages"]
            last_message = messages[-1]

            if node_name == "agent":

                if getattr(
                    last_message,
                    "tool_calls",
                    None
                ):

                    print("Requested tools:")

                    for tool_call in last_message.tool_calls:

                        print(
                            "Name:",
                            tool_call["name"]
                        )

                        print(
                            "Arguments:",
                            tool_call["args"]
                        )

            elif node_name == "tools":

                for message in messages:

                    print("\nTool:", message.name)
                    print("Result:")
                    print(message.content)

    # Read the final state saved by the checkpointer.
    saved_state = app.get_state(config)

    saved_messages = saved_state.values.get(
        "messages",
        []
    )

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    if (
        saved_messages
        and isinstance(saved_messages[-1], AIMessage)
    ):

        print(saved_messages[-1].content)

    else:

        print("No final answer generated.")


# ============================================================
# 10. MAIN APPLICATION
# ============================================================

def main():
    """Initialize and test the agent and its memory."""

    client, vs_collection, full_collection = init_mongodb()

    try:

        print("MongoDB Atlas connected successfully!")

        # Initialize the MongoDB tools.
        tools = create_tools(
            vs_collection,
            full_collection
        )

        tools_by_name = {
            current_tool.name: current_tool
            for current_tool in tools
        }

        # Initialize the local language model.
        llm = ChatOllama(
            model="llama3.2",
            temperature=0
        )

        print(
            "Ollama Llama 3.2 initialized successfully!"
        )

        # Tool-selection prompt.
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are an AI assistant specializing in MongoDB.

                Available tools: {tool_names}

                INSTRUCTIONS:

                - Use the vector search tool for general
                  technical questions about MongoDB.

                - Use the document retrieval tool when
                  the user asks for a page by its title.

                - If the user asks about the previous
                  conversation, answer using the
                  conversation history.

                - Do not use tools unnecessarily.

                - After answering a question, remember
                  the messages available in the conversation.

                - Do not invent missing information.

                - If you cannot answer from the available
                  information, say I DON'T KNOW.
                """
            ),
            MessagesPlaceholder(
                variable_name="messages"
            ),
        ])

        prompt = prompt.partial(
            tool_names=", ".join(
                current_tool.name
                for current_tool in tools
            )
        )

        # Give Llama access to the tools.
        llm_with_tools = (
            prompt
            | llm.bind_tools(tools)
        )

        # Separate prompt for generating the final answer
        # after retrieving MongoDB documentation.
        final_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are a technical assistant specializing
                in MongoDB.

                Answer the user's question using the
                retrieved documentation below.

                If a summary is requested, summarize
                the relevant document clearly.

                Use only the information provided
                in the retrieved documentation.

                Ignore documents unrelated to the question.

                Do not invent information.

                If the relevant information is missing,
                say I DON'T KNOW.

                Do not request additional tools.
                """
            ),
            (
                "human",
                "User question:\n{question}\n\n"
                "Retrieved documentation:\n{context}\n\n"
                "Provide your final answer:"
            ),
        ])

        final_chain = final_prompt | llm

        # Build LangGraph with persistent memory.
        app = init_graph(
            llm_with_tools,
            final_chain,
            tools_by_name,
            client
        )

        print(
            "LangGraph initialized "
            "with MongoDB memory!"
        )

        # Use the SAME thread ID for both questions.
        thread_id = "mongodb-training-memory-002"

        # TEST 1: Ask a question requiring vector search.
        execute_graph(
            app,
            thread_id,
            "What are some best practices "
            "for data backups in MongoDB?"
        )

        # TEST 2: Ask about the preceding conversation.
        execute_graph(
            app,
            thread_id,
            "What did I just ask you?"
        )

        # TEST 3: Inspect the conversation saved in MongoDB.
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        saved_state = app.get_state(config)

        print("\n" + "=" * 60)
        print("SAVED CONVERSATION")
        print("=" * 60)

        for message in saved_state.values["messages"]:

            if isinstance(message, HumanMessage):

                print(
                    "\nUSER:",
                    message.content
                )

            elif (
                isinstance(message, AIMessage)
                and not message.tool_calls
            ):

                print(
                    "\nAGENT:",
                    message.content
                )

        print("\nMemory test completed.")

    finally:

        client.close()


if __name__ == "__main__":
    main()
