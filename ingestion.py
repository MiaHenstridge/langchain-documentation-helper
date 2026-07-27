import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
import requests
import re

from langchain_community.document_loaders import SitemapLoader
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
    retry_min_seconds=60,
)

# local ChromaDB vector store (in case you want to use ChromaDB)
vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# Pinecone vector store
# vector_store = PineconeVectorStore(index_name=os.environ.get("INDEX_NAME"), embedding=embeddings)

# Tavily objects for crawling
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=100, max_pages=1000)
tavily_crawl = TavilyCrawl()



def chunk_urls(urls: List[str], chunk_size: int=20) -> List[List[str]]:
    """Split URLs into chunks of specified size"""
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i: i+chunk_size]
        chunks.append(chunk)
    return chunks


async def extract_batch(urls: List[str], batch_num:int) -> List[Dict[str, Any]]:
    """Extract documents from a batch of URLs"""
    try:
        log_info(
            f"TavilyExtract: Processing batch {batch_num} with {len(urls)} URLs",
            Colors.BLUE
        )
        docs = await tavily_extract.ainvoke(input={"urls": urls})
        log_success(
            f"TavilyExtract: Completed batch {batch_num} - extracted {len(docs.get('results', []))} documents",
            # Colors.GREEN
        )
        return docs
    except Exception as e:
        log_error(f"TavilyExtract: Failed to extract batch {batch_num} - {e}")
        return []


async def async_extract(url_batches: List[List[str]]): 
    log_header("DOCUMENT EXTRACTION PHASE")
    log_info(
        f"TavilyExtract: Starting concurrent extraction of {len(url_batches)} batches",
        Colors.DARKCYAN,
    )

    tasks = [extract_batch(batch, i+1) for i, batch in enumerate(url_batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and flatten result
    all_pages = []
    failed_batches = 0
    for result in results:
        if isinstance(result, Exception):
            log_error(f"TavilyExtract: Batch failed with exception - {result}")
            failed_batches += 1
        else:
            for extracted_page in result.get("results", []):
                document = Document(
                    page_content=extracted_page["raw_content"],
                    metadata={"source": extracted_page["url"]},
                )
                all_pages.append(document)
    
    log_success(
        f"TavilyExtract: Extraction complete: Total pages extracted {len(all_pages)}" 
    )
    if failed_batches > 0:
        log_warning(f"TavilyExtract: {failed_batches} batches failed during extraction")
    return all_pages


async def index_documents_async(documents: List[Document], batch_size: int=50):
    """Process documents in batches asynchronously"""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"VectorStore indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )
    # create batches
    batches = [
        documents[i:i+batch_size] for i in range(0, len(documents), batch_size)
    ]
    log_info(
        f"VectorStore indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )
    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num:int):
        try:
            await vector_store.aadd_documents(batch)
            log_success(
                f"VectorStore indexing: Successfully added batch {batch_num/len(batches)}: {len(batch)} documents"
            )
        except Exception as e:
            log_error(f"VectorStore indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    # process batches concurrently
    tasks = [add_batch(batch, i+1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for re in results if re is True)
    if successful == len(batches):
        log_success(
            f"VectorStore indexing: All batches processed successfully {successful}/{len(batches)}"
        )
    else:
        log_warning(
            f"VectorStore indexing: Processed {successful}/{len(batches)} batches successfully"
        )


async def main():
    """
    Main async function to orchestrate the entire pipeline
    """
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "TavilyCrawl: Starting to crawl documentation from https://docs.langchain.com/",
        Colors.PURPLE
    )

    # Initialize SitemapLoader to load URLs from the LangChain sitemap
    # Note: By default, SitemapLoader parses the page content too, but we pass 
    # a dummy parser to speed up the process since Tavily handles content extraction.
    sitemap_url = "https://docs.langchain.com/sitemap.xml"
    sitemap_loader = SitemapLoader(
        web_path=sitemap_url,
        filter_urls=[r"https://docs\.langchain\.com/oss/python/.*", r"https://docs\.langchain\.com/api-reference/.*"],
        parsing_function=lambda soup: "", # Skip parsing body HTML for faster loader performance
        max_depth=10,
    )
    
    # Load sitemap asynchronously
    sitemap_docs = await asyncio.to_thread(sitemap_loader.load)
    
    # Extract unique URLs from metadata
    all_urls = list({doc.metadata["source"] for doc in sitemap_docs if "source" in doc.metadata})
    
    print(f"Found {len(all_urls)} URLs from the sitemap")

    # split URLs into batches of 20
    url_batches = chunk_urls(all_urls, chunk_size=20)
    log_info(
        f"URL Processing: Split {len(all_urls)} URLs into {len(url_batches)} batches",
        Colors.BLUE
    )

    # extract documents in batches concurrently
    all_docs = await async_extract(url_batches)

    # split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=150)

    log_header("PIPELINE COMPLETE")
    log_success("Document ingestion pipeline finished successfully")
    log_info("Summary:", Colors.BOLD)
    log_info(f" URLs mapped: {len(all_urls)}")
    log_info(f" Documents extracted: {len(all_docs)}")
    log_info(f" Chunks created: {len(splitted_docs)}")

if __name__ == "__main__":
    asyncio.run(main())


    