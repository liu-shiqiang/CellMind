import logging
import os
import math
import concurrent.futures
import asyncio
from tqdm.asyncio import tqdm
from typing import Any, Dict, Iterable, List, Optional

import scanpy as sc

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.scripts.interpretation_types import RagTopic, RagTopicContext

from src.bio_pretrained_model.data_prep._scgpt_data_processor import ScGPTDataProcessor
from src.bio_pretrained_model.scgpt._scgpt_model import ScGPTModelWrapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class BioKnowledgeRag:
    def __init__(
            self,
            vector_store_path: str,
            embedding_model:str = "all-MiniLM-L6-v2",
            top_k: int = 5
    ):
        self.logger = logging.getLogger(__name__)
        self.vector_store_path = vector_store_path
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.embeddings = HuggingFaceEmbeddings(
            model_name='/home/share/huadjyin/home/liushiqiang/pretrained_model/all-MiniLM-L6-v2'
        )
        self.vector_stores: Dict[str, Chroma] = {}

    def init_vector_store(self,collection_name: str) -> None:
        """create or load vector store"""
        try:
            if collection_name in self.vector_stores:
                self.logger.info(f"Vector store {collection_name} already exists.")
                self.vector_store = self.vector_stores[collection_name]
                return self.vector_store
            
            if not os.path.exists(self.vector_store_path):
                os.makedirs(self.vector_store_path, exist_ok=True)

            self.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.vector_store_path,
                create_collection_if_not_exists=True
            )
            self.vector_stores[collection_name] = self.vector_store
            self.logger.info(f"Vector store '{collection_name}' initialized and cached.")
            return self.vector_store
        except Exception as e:
            self.logger.error(f"Initialization vector storage failed: {str(e)}")
            raise e


    def load_documents(self, dir_path: str):
        """load documents from directory"""
        try:
            self.logger.info(f"Load documents from directory: {dir_path}")

            loader = DirectoryLoader(
                path=dir_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs = {"encoding": "utf-8"},
                show_progress=True

            )
            return loader.load()
        except Exception as e:
            self.logger.error(f"Load documents failed: {str(e)}")
            raise
        

    async def add_literatures_documents(self,
                                  documents: List[Document], 
                                  batch_size: int = None, 
                                  max_batch_chars:int = 2_000_000,
                                  retry_limit:int = 2,
                                  max_workers:int = 4
                                  ) -> None:
        """split documents and add aplits to vector store"""
        try:
            self.logger.info(f"Add {len(documents)} documents to vector store")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            all_splits = text_splitter.split_documents(documents)
            total_splits = len(all_splits)
            self.logger.info(f"Add {len(all_splits)} splits to vector store")

            if not batch_size:
                avg_len = sum(len(doc.page_content) for doc in all_splits) / total_splits if total_splits else 1
                batch_size = max(1, int(max_batch_chars // avg_len))
                self.logger.info(f"Auto-estimated batch_size: {batch_size}")
            else:
                self.logger.info(f"Using batch_size: {batch_size}")

            batches = [all_splits[i:i+batch_size] for i in range(0,total_splits,batch_size)]
            total_batches = len(batches)

            sem = asyncio.Semaphore(max_workers)

            async def process_batch(batch_id, batch, attempt=1):
                async with sem:
                    try:
                        self.logger.info(f"[Batch {batch_id}] Try {attempt} with {len(batch)} splits")
                        await self.vector_store.aadd_documents(documents=batch)
                        return True
                    except Exception as e:
                        self.logger.error(f"[Batch {batch_id}] Failed on attempt {attempt}: {e}")
                        if attempt < retry_limit:
                            return await process_batch(batch_id, batch, attempt+1)
                        return False

            self.logger.info(f"Start uploading {total_batches} batches with max_workers={max_workers}")
            pbar = tqdm(total=total_batches, desc="Uploading to Vector Store")

            tasks = [process_batch(batch_id, batch) for batch_id, batch in enumerate(batches)]
            for future in asyncio.as_completed(tasks):
                await future
                pbar.update(1)

            pbar.close()

            self.logger.info("All splits added successfully")
        except Exception as e:
            self.logger.error(f"Add documents failed: {str(e)}",exc_info=True)
            raise
    

    def query(self,query:str, collection_name:str = None, top_k:int = None):
        """similarity search for query and return top_k results"""
        try:
            collection = self.vector_store if collection_name is None else self.vector_stores[collection_name]
            if collection is None:
                raise ValueError(f"Vector store{collection_name} is not initialized.")\
            
            top_k = top_k or self.top_k
            results = collection.similarity_search(query , k=top_k)
            return results
        except Exception as e:
            self.logger.error(f"Query failed: {str(e)}")
            raise e
    
    def rag_context_generate(self,query:str,collection_name:str = None,top_k:int = None):
        """rag: retrieve augment and generate"""
        try:
            docs = self.query(query,collection_name,top_k)
            context = "\n".join([doc.page_content for doc in docs])
            prompt = f"Answer the question based on the context below:\n\n{context}\n\nQuestion: {query}\nAnswer:"
            self.logger.info("Sending prompt to LLM...")
            return prompt
        except Exception as e:
            self.logger.error(f"RAG generation failed: {str(e)}")
            raise e

    async def interpret_topics(
        self,
        topics: List[RagTopic],
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None,
        max_concurrency: int = 4,
    ) -> List[RagTopicContext]:
        """Retrieve supporting documents for a batch of RAG topics."""

        if not topics:
            return []

        semaphore = asyncio.Semaphore(max(1, max_concurrency))
        loop = asyncio.get_running_loop()

        async def _fetch(topic: RagTopic) -> RagTopicContext:
            async with semaphore:
                docs: List[Document] = []
                used_query = topic.query_text
                queries = [topic.query_text]
                if topic.metadata:
                    alternates = topic.metadata.get("alternate_queries")
                    if isinstance(alternates, list):
                        queries.extend(str(item) for item in alternates if item)
                chosen_top_k = topic.metadata.get("top_k") if topic.metadata else None
                desired_top_k = int(chosen_top_k or top_k or self.top_k)

                for attempt, query in enumerate(queries, 1):
                    docs = await loop.run_in_executor(
                        None,
                        lambda q=query, k=desired_top_k: self.query(
                            q,
                            collection_name=collection_name,
                            top_k=k,
                        ),
                    )
                    if docs:
                        used_query = query
                        break

                combined = "\n\n".join(doc.page_content for doc in docs)
                metadata = {
                    "attempts": len(queries),
                    "used_query": used_query,
                    "top_k": desired_top_k,
                    "num_documents": len(docs),
                }
                if topic.metadata:
                    metadata.update({f"topic_{key}": value for key, value in topic.metadata.items()})

                return RagTopicContext(
                    topic=topic,
                    documents=docs,
                    combined_context=combined,
                    metadata=metadata,
                )

        tasks = [asyncio.create_task(_fetch(topic)) for topic in topics]
        results = await asyncio.gather(*tasks)
        return results

    def interpret_topics_sync(
        self,
        topics: List[RagTopic],
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None,
        max_concurrency: int = 4,
    ) -> List[RagTopicContext]:
        """Synchronous wrapper around :meth:`interpret_topics`."""

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():  # pragma: no cover - defensive branch
            raise RuntimeError(
                "interpret_topics_sync cannot be called while an event loop is running; "
                "await interpret_topics(...) instead."
            )

        return asyncio.run(
            self.interpret_topics(
                topics,
                collection_name=collection_name,
                top_k=top_k,
                max_concurrency=max_concurrency,
            )
        )



    
class CellRag:

    def __init__(self,chromadb_path:str,collection_name:str):
        
        self.client = chromadb.PersistentClient(path=chromadb_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add(self,adata_path: str):

        adata = sc.read_h5ad(adata_path)
        ids = [f"CIMA_{i}" for i in range(adata.n_obs)]
        metadatas = adata.obs[[
            "celltype_l1",
            "celltype_l2",
            "celltype_l3",
            "celltype_l4",
            "final_annotation",
            "cell_type_ontology_term_id",
            "sample",
            "batch",
            "n_genes_by_counts",
            "total_counts",
            "pct_counts_mt"
        ]].astype(str).to_dict(orient="records")

        embedding_key = "X_scgpt"
        embeddings = adata.obsm[embedding_key]
        try:
            batch_size = 5000
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                batch_metadata = metadatas[i:i + batch_size]
                
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings.tolist(),
                    metadatas=batch_metadata
                )
        except Exception as e:
            raise e

    def query(self, embedding: List[float], n_results: int = 5):

        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
            )
            return results
        except Exception as e:
            raise e

    def batch_add(self, data_dir: str):
        """batch add data to chroma db"""
        
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory {data_dir} does not exist.")
        for file in os.listdir(data_dir):
            if file.endswith(".h5ad"):
                file_path = os.path.join(data_dir, file)
                print(f"Loading {file_path}")

                try:
                    adata = sc.read_h5ad(file_path)

                    if "X_scgpt" not in adata.obsm or "scGPT_clusters" not in adata.obs.columns:
                        print(f"{file_path} has not undergone cell embedding processing.")
                        continue

                    self.add(adata)

                except Exception as e:
                    print (f"Error loading {file_path}: {e}")
        
        print("All the adata files have been added to chroma db.")

    def get_all_metadata(self):

        all = self.collection.get(include = ['metadatas'], limit = None)
        return all["metadatas"]


def find_similar_clusters(
    cell_rag: CellRag,
    embedding: Iterable[float],
    n_results: int = 5,
) -> List[Dict[str, Any]]:
    """Query the CellRag index and summarise the top matching cell populations."""

    if embedding is None:
        return []

    try:
        response = cell_rag.query(list(embedding), n_results=n_results)
    except Exception as exc:  # pragma: no cover - defensive guard
        logging.getLogger(__name__).warning("CellRag query failed: %s", exc)
        return []

    matches: List[Dict[str, Any]] = []
    ids = response.get("ids", [[]])[0] if isinstance(response.get("ids"), list) else []
    metadatas = response.get("metadatas", [[]])[0] if isinstance(response.get("metadatas"), list) else []
    distances = response.get("distances", [[]])[0] if isinstance(response.get("distances"), list) else []

    for idx, metadata in enumerate(metadatas):
        entry = {
            "reference_id": ids[idx] if idx < len(ids) else None,
            "metadata": metadata or {},
            "distance": distances[idx] if idx < len(distances) else None,
        }
        cell_type = None
        if isinstance(metadata, dict):
            for key in ("final_annotation", "celltype_l4", "celltype_l3", "celltype_l2", "celltype_l1"):
                if metadata.get(key):
                    cell_type = metadata[key]
                    break
        if cell_type:
            entry["cell_type"] = cell_type
        matches.append(entry)

    return matches
                









