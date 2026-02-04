#!/usr/bin/env python3
"""Test LLM annotation tool

Test the annotate_with_llm function on a sample dataset.
"""
import sys
import json
import scanpy as sc
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the underlying function directly, not the tool wrapper
from src.tools.annotation.llm_annotate import annotate_with_llm as annotate_with_llm_tool

# Get the underlying function from the tool
annotate_with_llm = annotate_with_llm_tool.func if hasattr(annotate_with_llm_tool, 'func') else annotate_with_llm_tool

# Use a dataset that already has clustering and marker genes
test_file = "/Users/alston/Documents/project/genomix-agent/runs/02649e77-7c0d-4844-b8a1-7707f4980a68/artifacts/data/marker_genes_markers_20260131_154400.h5ad"

print(f"Testing LLM annotation tool on: {test_file}")
print("=" * 60)

# First, let's check what's in the file
adata = sc.read_h5ad(test_file)
print(f"\nDataset info:")
print(f"  - n_obs: {adata.n_obs}")
print(f"  - n_vars: {adata.n_vars}")
print(f"  - obs columns: {list(adata.obs.columns)[:10]}...")
print(f"  - obsm keys: {list(adata.obsm.keys())}")
print(f"  - uns keys: {list(adata.uns.keys())}")

# Check for clustering
for key in ['leiden', 'louvain', 'clusters', 'cluster']:
    if key in adata.obs.columns:
        print(f"\nClustering key found: {key}")
        print(f"  - Unique values: {adata.obs[key].value_counts().to_dict()}")
        break

# Check for marker genes
if "rank_genes_groups" in adata.uns:
    print(f"\nMarker genes analysis found!")
    rgg = adata.uns['rank_genes_groups']
    print(f"  - Keys: {list(rgg.keys())}")
    if 'names' in rgg:
        names = rgg['names']
        print(f"  - Names type: {type(names)}")
        if hasattr(names, 'dtype') and names.dtype.names:
            print(f"  - Groups: {names.dtype.names}")

# Now test the annotation tool
print("\n" + "=" * 60)
print("Running annotate_with_llm...")
print("=" * 60)

# Test with use_rag=False first (faster, no RAG query)
# Use Ollama's qwen3:8b model
result_json = annotate_with_llm(
    file_path=test_file,
    use_rag=False,  # Disable RAG for faster testing
    llm_provider="ollama",
    llm_model="qwen3:8b",
    tissue_context="外周血单核细胞",
    species="human",
    save_result=False,
)

# Parse and display results
result = json.loads(result_json)

print("\n" + "=" * 60)
print("ANNOTATION RESULTS")
print("=" * 60)

print(f"\nStatus: {result['status']}")
print(f"Message: {result['message']}")

if result['status'] == 'success':
    print(f"\nMethod: {result['data'].get('method')}")
    print(f"Average Confidence: {result['data'].get('average_confidence', 0):.2f}")
    print(f"High Confidence Annotations: {result['data'].get('high_confidence_count', 0)}")

    print("\n" + "-" * 60)
    print("CLUSTER ANNOTATIONS:")
    print("-" * 60)

    for detail in result['data'].get('annotation_details', []):
        print(f"\nCluster {detail['cluster']}:")
        print(f"  Cell Type: {detail['cell_type']}")
        print(f"  Confidence: {detail['confidence']}")
        print(f"  Key Markers: {', '.join(detail.get('key_markers', [])[:5])}")
        print(f"  Method: {detail.get('method', 'unknown')}")
        if detail.get('reasoning'):
            print(f"  Reasoning: {detail['reasoning'][:150]}...")

    print("\n" + "-" * 60)
    print("CELL TYPE COUNTS:")
    print("-" * 60)
    for ct, count in result['data'].get('cell_type_counts', {}).items():
        print(f"  {ct}: {count}")

print("\n" + "=" * 60)
print("Test completed!")
print("=" * 60)
