FEWSHOT_EXAMPLES = [
        {
            "user": "Please annotate this h5ad file with cell types.",
            "assistant": (
                "Plan:\n"
                "1. Load the h5ad file.\n"
                "2. Extract embeddings using the scGPT model.\n"
                "3. Perform clustering and detect marker genes.\n"
                "4. Annotate clusters with a RAG-based method.\n"
                "5. Evaluate the annotation with ARI and NMI metrics.\n"
                "<END_OF_PLAN>"
            ),
        },
        {
            "user": "Help me analyse this single-cell dataset and report the results.",
            "assistant": (
                "Plan:\n"
                "1. Load the h5ad dataset.\n"
                "2. Generate 512-dimensional embeddings for each cell.\n"
                "3. Cluster cells and identify marker genes.\n"
                "4. Annotate clusters using an external knowledge base.\n"
                "5. Evaluate annotation accuracy and generate a summary.\n"
                "<END_OF_PLAN>"
            ),
        },
    ]