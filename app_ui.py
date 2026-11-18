import streamlit as st
import requests
import streamlit.components.v1 as components

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PathSnap Evaluator Dashboard",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ PathSnap: Distributed Map-Matching & Route Reconstruction")
st.markdown("**Evaluator-Facing Client Interface (Laptop 2)** — Connected to Laptop 1 backend.")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.title("⚙️ Deployment Settings")
laptop1_ip = st.sidebar.text_input(
    "Laptop 1 API URL", 
    value="http://10.1.34.91:8000", 
    help="Base URL of Laptop 1 running the FastAPI backend."
)

# Health Check & Trip Loading
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔌 Backend Status")

trips_list = []
try:
    health_resp = requests.get(f"{laptop1_ip.rstrip('/')}/health", timeout=3)
    if health_resp.status_code == 200:
        health_data = health_resp.json()
        st.sidebar.success(f"Connected! Model: {health_data.get('model', 'Unknown')}[cite: 4]")
    else:
        st.sidebar.warning("Backend returned non-200 status.")
except Exception:
    st.sidebar.error("Could not connect to Laptop 1 backend.")

# Fetch trips from /trips endpoint
try:
    trips_resp = requests.get(f"{laptop1_ip.rstrip('/')}/trips", timeout=3)
    if trips_resp.status_code == 200:
        trips_data = trips_resp.json()
        trips_list = trips_data.get("trips", [])
except Exception:
    pass

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Evaluation Trips")

if trips_list:
    trip_options = {t["trip_id"]: f"Trip {t['trip_id']} ({t.get('n_obs', '?')} obs)" for t in trips_list}
    selected_trip_id = st.sidebar.selectbox("Select Trip ID", list(trip_options.keys()), format_func=lambda x: trip_options[x])
else:
    # Fallback list matching the 10 official evaluator test trips
    fallback_ids = ["00000015", "00000017", "00000023", "00000030", "00000049", "00000059", "00000067", "00000073", "00000087", "00000094"]
    selected_trip_id = st.sidebar.selectbox("Select Trip ID (Fallback)", fallback_ids)

run_button = st.sidebar.button("🚀 Run /match Pipeline", type="primary", use_container_width=True)

# --- MAIN DASHBOARD LAYOUT ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Performance Metrics")
    metrics_container = st.empty()
    metrics_container.info("Select a trip and click **Run /match Pipeline** in the sidebar.")

with col2:
    st.subheader("🗺️ Reconstructed Route Map")
    map_container = st.empty()
    map_container.info("Map visualization will appear here after running inference.")

# --- SERVER LOGS PANEL ---
st.markdown("---")
with st.expander("📜 Live Server Logs (Laptop 1)", expanded=False):
    if st.button("Fetch Recent Logs"):
        try:
            logs_resp = requests.get(f"{laptop1_ip.rstrip('/')}/logs?n=30", timeout=3)
            if logs_resp.status_code == 200:
                lines = logs_resp.json().get("lines", [])
                st.code("\n".join(lines), language="text")
            else:
                st.warning("Failed to fetch logs.")
        except Exception as e:
            st.error(f"Error connecting for logs: {e}")

# --- EXECUTION LOGIC ---
if run_button:
    api_endpoint = f"{laptop1_ip.rstrip('/')}/match"
    payload = {"trip_id": selected_trip_id}
    
    with st.spinner(f"Executing live pipeline on Laptop 1 for Trip {selected_trip_id}..."):
        try:
            response = requests.post(api_endpoint, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                st.success(f"Inference completed successfully in {data.get('latency_ms') or 0:.1f} ms!")
                
                # Extract metrics safely with defaults
                metrics = data.get("metrics") or {}
                classifier_top1 = data.get("classifier_top1") or 0.0
                hmm_top1 = data.get("hmm_top1") or 0.0
                latency = data.get("latency_ms") or 0.0
                
                # Render Metrics Panel safely
                with col1:
                    metrics_container.markdown(f"""
                    - **Trip ID:** `{data.get('trip_id')}`[cite: 4]
                    - **Observations:** `{data.get('n_obs')}`[cite: 4]
                    - **Duration:** `{data.get('duration_s') or 0:.1f}s`[cite: 4]
                    - **Latency:** `{latency:.1f} ms`[cite: 4]
                    - **Classifier Top-1:** `{classifier_top1:.3f}`[cite: 4]
                    - **HMM Top-1:** `{hmm_top1:.3f}`[cite: 4]
                    - **Top-1 Accuracy:** `{metrics.get('top1_accuracy') or 0:.3f}`[cite: 4]
                    - **Macro F1:** `{metrics.get('macro_f1') or 0:.3f}`[cite: 4]
                    - **Route Continuity (Strict):** `{metrics.get('route_continuity_strict') or 0:.3f}`[cite: 4]
                    - **Route Continuity (Relaxed):** `{metrics.get('route_continuity_relaxed') or 0:.3f}`[cite: 4]
                    - **LCS F1:** `{metrics.get('lcs_f1') or 0:.3f}`[cite: 4]
                    - **Length F1:** `{metrics.get('len_f1') or 0:.3f}`[cite: 4]
                    """)
                
                # Render Map in Iframe
                map_html = data.get("map_html", "")
                if map_html:
                    with col2:
                        map_container.empty()
                        components.html(map_html, height=550, scrolling=True)
                else:
                    with col2:
                        map_container.warning("No map HTML returned by backend.")
                        
            elif response.status_code == 400:
                err = response.json().get("detail", "Bad Request")
                st.error(f"Error 400: {err}[cite: 4]")
            elif response.status_code == 404:
                err = response.json().get("detail", "Not Found")
                st.error(f"Error 404: {err}[cite: 4]")
            else:
                st.error(f"Server Error ({response.status_code}): {response.text}[cite: 4]")
                
        except requests.exceptions.ConnectionError:
            st.error(f"❌ Could not connect to Laptop 1 backend at `{api_endpoint}`. Check network or IP address[cite: 4].")
        except requests.exceptions.Timeout:
            st.error("⏳ Request timed out. The server took too long to respond.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")