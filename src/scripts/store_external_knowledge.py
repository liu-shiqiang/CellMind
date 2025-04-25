import asyncio
from src.scripts.rag import BioKnowledgeRAG, ChromaDB
from config.setting import settings

async def store_literature_knowledge(lit_path, vector_store_path, collection_name):
    
    bioknowledgerag = BioKnowledgeRAG(vector_store_path)
    bioknowledgerag.init_vector_store(collection_name)
    Documents = bioknowledgerag.load_documents(lit_path)
    await bioknowledgerag.add_literatures_documents(Documents)

    return None
    

def store_singlecell_data(chroma_path, collection_name, adata_path):

    chromadb = ChromaDB(chroma_path, collection_name)
    chromadb.batch_add(adata_path)

    return None

    


if __name__ == "__main__":
    
    asyncio.run(store_literature_knowledge(lit_path=settings.literature_path,
                               vector_store_path=settings.CHROMADB_PERSIST_DIR,
                               collection_name='lit_rag'))
    
    print('Literature knowledge stored')