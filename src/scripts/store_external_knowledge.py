import asyncio
from src.scripts.rag import BioKnowledgeRag,CellRag
from config.setting import settings
from langchain_chroma import Chroma

async def store_literature_knowledge(lit_path, vector_store_path, collection_name):
    
    bioknowledgerag = BioKnowledgeRag(vector_store_path)
    bioknowledgerag.init_vector_store(collection_name)
    Documents = bioknowledgerag.load_documents(lit_path)
    await bioknowledgerag.add_literatures_documents(Documents)

    return None


def store_singlecell_data(chroma_path, collection_name, adata_path):

    cellrag = CellRag(chroma_path, collection_name)
    cellrag.add(adata_path)
    print(f"Single-cell data stored in collection: {collection_name}")

    return None

def collection_count(chroma_path, collection_name):

    bioknowledgerag = BioKnowledgeRag(chroma_path)
    bioknowledgerag.init_vector_store(collection_name)
    return bioknowledgerag.vector_store._chroma_collection.count()
    
def query(chroma_path, collection_name, query):

    bioknowledgerag = BioKnowledgeRag(chroma_path)
    bioknowledgerag.init_vector_store(collection_name)
    result = bioknowledgerag.rag_context_generate(query)
    return result

if __name__ == "__main__":
    
    # asyncio.run(store_literature_knowledge(lit_path=settings.LITERATURE_PATH,
    #                            vector_store_path=settings.CHROMADB_PERSIST_DIR,
    #                            collection_name=settings.CHROMADB_lit_collection_name))


    # count = collection_count(chroma_path=settings.CHROMADB_PERSIST_DIR,
    #                         collection_name=settings.CHROMADB_lit_collection_name)

    # question = "What are the marker genes of B cells?"
    # result = query(chroma_path=settings.CHROMADB_PERSIST_DIR,
    #                 collection_name=settings.CHROMADB_lit_collection_name,
    #                 query=question)
    # print(result)
    adata_path = "/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/output/cima_train_vector_db_with_embedding.h5ad"
    store_singlecell_data(chroma_path=settings.CHROMADB_PERSIST_DIR,
                            collection_name=settings.CHROMADB_cell_collection_name,
                            adata_path=adata_path)

    
