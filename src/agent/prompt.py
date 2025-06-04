FEWSHOT_EXAMPLES = [
        {
            "user": "Please annotate this h5ad file with cell types.The data_path is:/home/share/huadjyin/home/liushiqiang/Projects/Blada/data/scgpt/cell_anno/ms/c_data.h5ad",
            "assistant": (
                "Plan:\n"
                "1. Load and process the h5ad file.\n"
                "2. Extract embeddings from preprocessed file using the scGPT model.\n"
                "3. Cluster the embeddings.\n"
                "4. Annotate clusters with a RAG-based method.\n"
                "<END_OF_PLAN>"
            ),
        }
    ]

# "5. Evaluate the annotation with ARI and NMI metrics.\n"