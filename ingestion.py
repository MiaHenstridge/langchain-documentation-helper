import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import Colors, log_error, log_header, log_info, log_success, log_warning

load_dotenv()


# Configure SSL context to use certifi certificates
ssl_connect = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10,
)

# local ChromaDB vector store (in case you want to use ChromaDB)
# chroma = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# Pinecone vector store
vector_store = PineconeVectorStore(index_name=os.environ.get("INDEX_NAME"), embedding=embeddings)

# Tavily objects for crawling
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def main():
    """
    Main async function to orchestrate the entire pipeline
    """
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "   TavilyCrawl: Starting to crawl documentation from https://docs.langchain.com/oss/python/langchain/overview",
        Colors.PURPLE
    )

    # crawl the documentation site
    res = tavily_crawl.invoke({
        "url": "https://reference.langchain.com/python/langchain/overview/",
        "max_depth": 5,
        "extract_depth": "advanced",
        "instructions": "content on ai agents",
    })
    all_docs = [Document(page_content=result['content'], metadata={"source": result['url']}) for result in res["results"]]
    log_success(
        f"TavilyCrawl: Successfully crawled {len(all_docs)} URLs from documentation site"
    )

    log_info()

if __name__ == "__main__":
    asyncio.run(main())


    