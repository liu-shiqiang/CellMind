# ui/app.py

import streamlit as st
import requests
import json
import os
import time
from typing import Optional


API_BASE_URL = os.environ.get("GENOMIX_AGENT_API", "http://localhost:8000")

# Legacy endpoints
RUN_ENDPOINT = f"{API_BASE_URL}/api/v2/agent/run"
STREAM_ENDPOINT = f"{API_BASE_URL}/api/v2/agent/stream"
RUN_AND_STREAM_ENDPOINT = f"{API_BASE_URL}/api/v2/agent/run-and-stream"

# Job-based endpoints
JOBS_ENDPOINT = f"{API_BASE_URL}/jobs"
JOB_EVENTS_ENDPOINT = f"{API_BASE_URL}/jobs/{{job_id}}/events"

st.set_page_config(page_title="🧬 Genomix Agent", layout="wide")

st.title("🧬 Genomix Agent - AI-Powered Multi-Omics Analysis")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Mode toggle: legacy vs job-based
    use_job_mode = st.toggle(
        "🔄 Use Job-Based API",
        value=False,
        help=(
            "OFF: Legacy agent streaming mode\n"
            "ON: New job-based API with SSE streaming\n\n"
            "Job mode provides better progress tracking and reconnection support."
        ),
    )

    stream_mode = st.selectbox(
        "Streaming Mode",
        options=["updates", "messages", "debug"],
        index=0,
        help=(
            "updates: Show progress updates (default)\n"
            "messages: Stream LLM tokens in real-time\n"
            "debug: Show detailed execution info"
        ),
    )

    st.markdown("---")
    st.markdown("### 💡 Tips")
    if use_job_mode:
        st.markdown(
            """
            - **Job mode**: Enter local dataset path directly
            - **updates mode**: Best for tracking progress
            - **messages mode**: Best for watching AI generate text
            - Enter your analysis objective and click run
            """
        )
    else:
        st.markdown(
            """
            - **Legacy mode**: Upload .h5ad file for analysis
            - **updates mode**: Best for tracking progress
            - **messages mode**: Best for watching AI generate text
            - Click run and watch the progress!
            """
        )

# Main interface
col1, col2 = st.columns([1, 1])

# Initialize session state for job mode
if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

if use_job_mode:
    # === JOB MODE: File upload + Create Job + Upload File buttons ===
    with col1:
        st.subheader("📁 Data Input")

        # File uploader for job mode
        job_uploaded_file = st.file_uploader(
            "Upload data file",
            type=["h5ad", "mtx", "tsv", "csv"],
            help="Upload your data file (.h5ad, .mtx, .tsv, .csv)",
            key="job_mode_uploader",
        )

        # Display current job info if exists
        if st.session_state.job_id:
            st.info(f"📋 Job ID: `{st.session_state.job_id[:8]}...`")

        # Display uploaded file info
        if st.session_state.uploaded_filename:
            st.success(f"✅ Uploaded: `{st.session_state.uploaded_filename}`")

    with col2:
        st.subheader("🎯 Analysis Objective")

        # User input
        objective_text = st.text_area(
            "Enter analysis objective",
            placeholder="e.g., 'Perform cell type annotation on the data'",
            height=80,
            help="Describe what you want to analyze",
        )

        # Create Job button
        create_job_button = st.button(
            "📋 Create Job",
            type="secondary",
            use_container_width=True,
        )

        # Upload File button (only enabled after job is created)
        upload_button = st.button(
            "⬆️ Upload File",
            type="secondary",
            use_container_width=True,
            disabled=st.session_state.job_id is None,
        )

        # Run Analysis button (only enabled after file is uploaded)
        run_button = st.button(
            "🚀 Run Analysis",
            type="primary",
            use_container_width=True,
            disabled=(st.session_state.job_id is None or
                     st.session_state.uploaded_filename is None),
        )

    # Handle Create Job button click
    if create_job_button:
        if not objective_text.strip():
            st.warning("⚠️ Please enter an analysis objective.")
        else:
            with st.spinner("Creating job..."):
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/jobs",
                        json={
                            "objective": objective_text.strip(),
                            "input_files": [],
                            "stream_mode": stream_mode,
                        },
                        timeout=30,
                    )
                    resp.raise_for_status()
                    job_data = resp.json()
                    st.session_state.job_id = job_data["job_id"]
                    st.session_state.uploaded_filename = None  # Reset
                    st.success(f"✅ Job created: `{st.session_state.job_id[:8]}...`")
                    st.rerun()
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Failed to create job: {e}")

    # Handle Upload File button click
    if upload_button:
        if st.session_state.job_id is None:
            st.error("⚠️ Please create a job first.")
        elif job_uploaded_file is None:
            st.warning("⚠️ Please select a file to upload.")
        else:
            with st.spinner("Uploading file..."):
                try:
                    files = {"file": (job_uploaded_file.name, job_uploaded_file, job_uploaded_file.type)}
                    resp = requests.post(
                        f"{API_BASE_URL}/jobs/{st.session_state.job_id}/upload",
                        files=files,
                        timeout=300,  # 5 minutes for large files
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    st.session_state.uploaded_filename = result["filename"]
                    st.success(f"✅ Uploaded: `{result['filename']}`")
                    st.rerun()
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Upload failed: {e}")

else:
    # === LEGACY MODE: Original file upload + Run button ===
    with col1:
        st.subheader("📁 Data Input")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload .h5ad file",
            type=["h5ad"],
            help="Upload your single-cell RNA-seq data file",
        )

    with col2:
        st.subheader("🎯 Analysis Objective")

        # User input
        objective_text = st.text_area(
            "Enter analysis objective",
            placeholder="e.g., 'Perform cell type annotation on the data'",
            height=120,
            help="Describe what you want to analyze",
        )

    # Run button
    run_button = st.button(
        "🚀 Run Analysis",
        type="primary",
        use_container_width=True,
    )


def process_sse_event(event: dict, containers: dict, session_state: dict) -> bool:
    """Process a single SSE event.

    Returns True if execution should continue, False if ended/error.
    """
    event_type = event.get("type", "unknown")
    payload = event.get("payload", {})
    node = event.get("node", "unknown")

    status_placeholder = containers["status"]
    progress_container = containers["progress"]
    output_container = containers["output"]
    result_container = containers["result"]
    progress_bar = containers["progress_bar"]

    # Handle different event types
    if event_type == "start":
        status_placeholder.empty()
        progress_container.write(f"🚀 {payload.get('message', 'Starting...')}")
        progress_bar.progress(5, text="Initializing...")

    elif event_type == "node_enter":
        message = payload.get("message", f"Executing: {node}")
        progress_container.write(f"⚙️  {message}")
        session_state["node_count"] += 1
        progress = min(5 + session_state["node_count"] * 5, 90)
        progress_bar.progress(progress, text=message)
        session_state["last_node"] = node

    elif event_type == "plan_update":
        message = payload.get("message", "Plan updated")
        plan = payload.get("plan", [])
        progress_container.info(f"📋 {message}")
        with progress_container.expander("View Plan Steps"):
            for i, step in enumerate(plan, 1):
                st.write(f"{i}. {step}")

    elif event_type == "tool_call":
        message = payload.get("message", "Calling tools")
        progress_container.write(f"🔧 {message}")

    elif event_type == "tool_result":
        message = payload.get("message", "Tool result received")
        progress_container.success(f"✅ {message}")

    elif event_type == "token":
        # Real-time token streaming
        token = payload.get("token", "")
        session_state["accumulated_text"] += token

        # Display accumulated text with streaming effect
        with output_container:
            st.subheader("📝 AI Response")
            st.markdown(session_state["accumulated_text"])

    elif event_type == "progress":
        # Progress update event
        progress_pct = payload.get("progress", 0)
        message = payload.get("message", "Processing...")
        progress_bar.progress(progress_pct, text=message)

    elif event_type == "heartbeat":
        # Keep connection alive - show subtle indicator
        with status_placeholder:
            st.caption("📡 Connected...")

    elif event_type == "end":
        # Execution complete
        progress_bar.progress(100, text="Complete!")
        message = payload.get("message", "Analysis complete!")
        with status_placeholder:
            st.success(message)

        # Check for final response
        final_response = payload.get("response")
        if final_response:
            with result_container:
                st.subheader("📊 Final Result")
                if isinstance(final_response, dict):
                    content = final_response.get("content", "")
                    if content:
                        st.markdown(content)
                    else:
                        st.json(final_response)
                else:
                    st.write(final_response)

        # Check for errors
        error = payload.get("error")
        if error:
            with result_container:
                st.error("❌ Execution failed")
                st.json(error)

        return False  # Stop processing

    elif event_type == "error":
        # Error occurred
        progress_bar.progress(0, text="Error!")
        detail = payload.get("detail", "Unknown error")
        status_placeholder.error(f"❌ Error: {detail}")
        return False  # Stop processing

    else:
        # Unknown event type - show in debug mode
        if session_state.get("stream_mode") == "debug":
            with output_container:
                with st.expander(f"Debug: {event_type}"):
                    st.json(event)

    return True  # Continue processing


if run_button:
    # Validate inputs based on mode
    if use_job_mode:
        # Job Mode: validate job_id exists and file is uploaded
        if st.session_state.job_id is None or st.session_state.uploaded_filename is None:
            st.warning("⚠️ Please create a job and upload a file first.")
            st.stop()
    else:
        # Legacy Mode: validate file upload and objective
        if uploaded_file is None or not objective_text.strip():
            st.warning("⚠️ Please upload a .h5ad file and enter your objective.")
            st.stop()

    # Initialize session state for streaming
    if "events" not in st.session_state:
        st.session_state.events = []
    if "accumulated_text" not in st.session_state:
        st.session_state.accumulated_text = ""

    st.session_state.events = []
    st.session_state.accumulated_text = ""
    st.session_state.stream_mode = stream_mode

    # Create containers
    status_placeholder = st.container()
    progress_container = st.container()
    output_container = st.container()
    result_container = st.container()

    containers = {
        "status": status_placeholder,
        "progress": progress_container,
        "output": output_container,
        "result": result_container,
    }

    session_state = {
        "accumulated_text": "",
        "node_count": 0,
        "last_node": None,
        "stream_mode": stream_mode,
    }

    with status_placeholder:
        st.info("📡 Initializing...")

    try:
        if use_job_mode:
            # ===== JOB-BASED MODE =====
            # Use existing job_id from session state
            job_id = st.session_state.job_id

            # Step 1: Start job execution
            with status_placeholder:
                st.info(f"📡 Starting job {job_id[:8]}...")

            run_response = requests.post(
                f"{API_BASE_URL}/jobs/{job_id}/run",
                timeout=30,
            )
            run_response.raise_for_status()

            # Step 2: Stream events via SSE (from beginning to include upload events)
            progress_bar = progress_container.progress(0, text="Starting...")

            with requests.get(
                JOB_EVENTS_ENDPOINT.format(job_id=job_id),
                params={"from": 0},
                stream=True,
                timeout=600,
            ) as response:
                response.raise_for_status()

                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    if not raw_line.startswith("data:"):
                        continue

                    data_str = raw_line[5:].strip()
                    if not data_str:
                        continue

                    try:
                        event = json.loads(data_str)
                        st.session_state.events.append(event)

                        # Add progress_bar to containers for event processing
                        containers["progress_bar"] = progress_bar

                        should_continue = process_sse_event(event, containers, session_state)
                        if not should_continue:
                            break

                    except json.JSONDecodeError as e:
                        st.warning(f"Failed to parse event: {e}")
                        continue

        else:
            # ===== LEGACY MODE =====
            # Save uploaded file
            save_dir = "./uploaded_data"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.read())

            # Build objective
            objective = f"{objective_text.strip()}, data_path:{os.path.abspath(save_path)}"

            # Use run-and-stream endpoint for simplicity
            with requests.post(
                RUN_AND_STREAM_ENDPOINT,
                params={"stream_mode": stream_mode},
                json={
                    "objective": objective,
                    "input_files": [os.path.abspath(save_path)],
                },
                stream=True,
                timeout=600,
            ) as response:
                response.raise_for_status()

                # Initialize progress tracking
                progress_bar = progress_container.progress(0, text="Starting...")

                # Process SSE stream
                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    if not raw_line.startswith("data:"):
                        continue

                    data_str = raw_line[5:].strip()
                    if not data_str:
                        continue

                    try:
                        event = json.loads(data_str)
                        st.session_state.events.append(event)

                        # Add progress_bar to containers for event processing
                        containers["progress_bar"] = progress_bar

                        should_continue = process_sse_event(event, containers, session_state)
                        if not should_continue:
                            break

                    except json.JSONDecodeError as e:
                        st.warning(f"Failed to parse event: {e}")
                        continue

    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. The analysis might take longer than expected.")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API call failed: {e}")

    # Display output files if they exist (legacy mode only)
    if not use_job_mode:
        result_container.markdown("---")
        result_container.subheader("📁 Output Files")

        cluster_map_path = os.path.join(os.path.dirname(save_path), "cluster_celltype_map.json")
        umap_image_path = os.path.join(os.path.dirname(save_path), "figures", "umap_llm_celltypes.png")

        if os.path.exists(cluster_map_path):
            with result_container.expander("📄 Cluster Celltype Map"):
                with open(cluster_map_path, "r") as f:
                    cluster_data = json.load(f)
                st.json(cluster_data)
                st.download_button(
                    "Download JSON",
                    data=json.dumps(cluster_data, indent=2),
                    file_name="cluster_celltype_map.json",
                    mime="application/json",
                )

        if os.path.exists(umap_image_path):
            result_container.subheader("🖼️ UMAP Visualization")
            result_container.image(umap_image_path, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <small>Powered by LangGraph | Built with ❤️ for multi-omics analysis</small>
    </div>
    """,
    unsafe_allow_html=True,
)
