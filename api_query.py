# %%
import nest_asyncio
nest_asyncio.apply()

from llama_index import download_loader
download_loader("GithubRepositoryReader")

from dotenv import load_dotenv

load_dotenv()

import os
import logging
import sys
import pickle
from pathlib import Path

import faiss

from llama_index import (
    VectorStoreIndex,
    load_index_from_storage,
    get_response_synthesizer,
    StorageContext,
    ServiceContext
)
from llama_index.llms import OpenAI
from llama_hub.github_repo import GithubClient, GithubRepositoryReader
from llama_index.vector_stores.faiss import FaissVectorStore

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

DATA_PATH = Path("./data")

llm = OpenAI(temperature=0.1, model="gpt-4")
service_context = ServiceContext.from_defaults(llm=llm)

def load_documents():
    documents_pkl_path = DATA_PATH / "local-api-documents.pkl"
    if os.path.exists(documents_pkl_path):
        with open(documents_pkl_path, "rb") as f:
            return pickle.load(f)

    github_client = GithubClient()

    loader = GithubRepositoryReader(
        github_client,
        owner =                  "gpsinsight",
        repo =                   "api-v2-documentation",
        filter_file_extensions = ([".md"], GithubRepositoryReader.FilterType.INCLUDE),
        verbose =                True,
        concurrent_requests =    1,
    )

    documents = loader.load_data(branch="master")

    with open(documents_pkl_path, "wb") as f:
        pickle.dump(documents, f)

    return documents


def get_or_build_index():

    faiss_index = faiss.IndexFlatL2(1536)

    index_path = DATA_PATH / "local-api-index"
    if os.path.exists(index_path):
        vector_store = FaissVectorStore.from_persist_dir(index_path)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=index_path
        )
        return load_index_from_storage(storage_context=storage_context)

    documents = load_documents()

    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context, 
        service_context=service_context
    )
    index.storage_context.persist(DATA_PATH / "local-api-index")
    return index

index = get_or_build_index()

response_synthesizer = get_response_synthesizer(
    service_context=service_context,
    response_mode="compact",
    verbose=True,
)

query_engine = index.as_query_engine(
    response_synthesizer=response_synthesizer
)

def qa(question: str):
    print(f"\033[94mQ: {question}\033[0m")

    response = query_engine.query(question)

    print(f"\033[92mA: {response}\033[0m")


# %%

qa("Can you please write code to get all alerts? In python.")


# %%

qa("Please write code to get alerts from last 40 days. In python.")

# %%
