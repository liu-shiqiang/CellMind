

class Settings:
       
    LOCAL_MODEL_PATH: str = "/home/share/huadjyin/home/liushiqiang/pretrained_model/"
    CHROMADB_PERSIST_DIR: str = "/home/share/huadjyin/home/liushiqiang/Projects/Blada/chroma_data"
    OUTPUT_DIR = "/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/output"
    SCGPT_MODEL_DIR = "/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/save/scgpt/scgpt-model"
    MARKER_REFERENCE_PATH: str = "/home/share/huadjyin/home/liushiqiang/Projects/Blada/data/cell_marker_all.csv"
    literature_path: str = "/home/share/huadjyin/home/liushiqiang/Projects/Blada/data/literature_knowledge_base"
    singlecell_path: str = "/home/share/huadjyin/home/liushiqiang/Projects/Blada/data/cell_type/immune"
    GENEFORMER_MODEL_PATH: str = "/models/geneformer"


settings = Settings() 