

class Settings:
       
    LOCAL_MODEL_PATH: str = "/home/share/huadjyin/home/liushiqiang/pretrained_model/"
    CHROMADB_PERSIST_DIR: str = "/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/chroma_data"
    CHROMADB_cell_collection_name: str = "cell_rag"
    CHROMADB_lit_collection_name: str = "lit_rag"
    LITERATURE_PATH: str = "/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/data/literature_knowledge_base"
    OUTPUT_DIR = "/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/output"
    SCGPT_MODEL_DIR = "/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/save/scgpt/scgpt-model"
    MARKER_REFERENCE_PATH: str = "/home/share/huadjyin/home/liushiqiang/Projects/Blada/data/cell_marker_all.csv"
    singlecell_path: str = "/home/share/huadjyin/home/liushiqiang/Projects/Blada/data/cell_type/immune"
    GENEFORMER_MODEL_PATH: str = "/models/geneformer"
    RETRIVE_TOP_K:int = 3
    NEO4J_URI = "neo4j+s://c0651eec.databases.neo4j.io"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "eWRgS3xons7xBhxaoZM0fr1SJZeANZoS6d_334ykH1k"
    NEO4J_DATABASE = "neo4j"


settings = Settings() 