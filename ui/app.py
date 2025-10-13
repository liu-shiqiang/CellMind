# ui/app.py

import streamlit as st
import requests
import json
import os


API_BASE_URL = os.environ.get("GENOMIX_AGENT_API", "http://localhost:8000")
RUN_ENDPOINT = f"{API_BASE_URL}/api/v2/agent/run"
STREAM_ENDPOINT = f"{API_BASE_URL}/api/v2/agent/stream"

st.set_page_config(page_title="🧬 Cell Annotation Agent", layout="wide")

st.title("🧬 genomix Agent")

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
                response = requests.post(
                    RUN_ENDPOINT,
                    json={
                        "objective": objective,
                        "input_files": [os.path.abspath(save_path)],
                    },
                    timeout=30,
                )
                response.raise_for_status()
                run_info = response.json()
            except Exception as e:
                st.error(f"API call failed: {e}")
                st.stop()

        run_id = run_info.get("run_id")
        if not run_id:
            st.error("Backend did not return a run_id")
            st.stop()

        st.subheader("🧠 Agent Events")
        event_container = st.container()
        event_log = []
        final_response = None
        error_payload = None

        try:
            with requests.get(f"{STREAM_ENDPOINT}/{run_id}", stream=True, timeout=300) as stream_resp:
                stream_resp.raise_for_status()
                for raw_line in stream_resp.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    if not raw_line.startswith("data:"):
                        continue
                    data_str = raw_line[5:].strip()
                    if not data_str:
                        continue
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    event_log.append(event)
                    event_container.json(event)
                    if event.get("type") == "end":
                        payload = event.get("payload", {})
                        final_response = payload.get("response")
                        error_payload = payload.get("error")
                        break
        except Exception as e:
            st.error(f"Streaming failed: {e}")
            st.stop()

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

        if error_payload:
            st.error("❌ Task failed")
            st.json(error_payload)
        else:
            st.success("✅ Annotation completed!")
            if final_response:
                st.subheader("🧾 Final Response")
                st.json(final_response)
