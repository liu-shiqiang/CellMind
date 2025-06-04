# ui/app.py

import streamlit as st
import requests
import os

API_URL = "http://localhost:8000/api/run"  # 若部署在远程服务器，请替换 IP

st.set_page_config(page_title="🧬 Cell Annotation Agent", layout="wide")

st.title("🧬 Cell Annotation Agent")

# Upload h5ad file
uploaded_file = st.file_uploader("Upload .h5ad file", type=["h5ad"])

# User input
objective_text = st.text_area("Enter analysis objective (you can mention the file too):")

if st.button("Run Agent"):

    if uploaded_file is None or not objective_text.strip():
        st.warning("Please upload a .h5ad file and enter your objective.")
    else:
        # Save uploaded file to disk
        save_dir = "./uploaded_data"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())

        # Build full objective text
        objective = f"{objective_text.strip()}, data_path:{os.path.abspath(save_path)}"
        st.success("File saved. Sending to agent...")
        st.code(objective)

        # Call backend API
        with st.spinner("Agent is working..."):
            try:
                response = requests.post(API_URL, json={"objective": objective})
                response.raise_for_status()
                result = response.json()
            except Exception as e:
                st.error(f"API call failed: {e}")
                st.stop()

        # Show agent events
        st.subheader("🧠 Agent Events")
        for event in result.get("events", []):
            st.json(event)

        # Optional: check for output files (umap, json)
        cluster_map_path = os.path.join(os.path.dirname(save_path), "cluster_celltype_map.json")
        umap_image_path = os.path.join(os.path.dirname(save_path), "figures", "umap_llm_celltypes.png")

        if os.path.exists(cluster_map_path):
            st.subheader("📄 Cluster Celltype Map")
            with open(cluster_map_path, "r") as f:
                st.json(f.read())

        if os.path.exists(umap_image_path):
            st.subheader("🖼️ UMAP Visualization")
            st.image(umap_image_path)

        st.success("✅ Annotation completed!")
