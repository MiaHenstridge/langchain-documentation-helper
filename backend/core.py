import os
from typing import Any, Dict

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


load_dotenv()

# Initialize embeddings (same as ingestion.py)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize vector store
vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# Intialize chat model
model = init_chat_model(model="gpt-5.2", model_provider="openai")


# tool function
@tool(response_format='content_and_artifact')       # this tool will return 2 values
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about Langchain"""
    # retrieve top 4 most relevant documents
    retrieved_docs = vector_store.as_retriever().invoke(query, k=4)
    # serialize documents for the model
    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    # return both serialized content and raw documents
    return serialized, retrieved_docs


# create agent
def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation

    Args:
        - query: The user's question
    
    Returns:
        Dictionary containing
            - answer: The generated answer
            - context: List of retrieved documents
    """
    # create the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )
    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)
    # build messages list
    messages = [{
        "role": "user",
        "content": query,
    }]
    # invoke the agent
    response = agent.invoke({"messages": messages})
    # extract the answer from the last AI message
    answer = response["messages"][-1].content
    # extract the context documents from ToolMessage artifacts
    context_docs = []
    for msg in response["messages"]:
        # check if this is a ToolMessage with artifact
        if isinstance(msg, ToolMessage) and hasattr(msg, "artifact"):
            # the artifact should contain the list of Document objects
            if isinstance(msg.artifact, list):
                context_docs.extend(msg.artifact)
    return {
        "answer": answer,
        "context": context_docs,
    }


if __name__ == "__main__":
    result = run_llm(query="What are deep agents?")
    print(result)