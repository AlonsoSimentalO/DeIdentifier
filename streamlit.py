import streamlit as st
import subprocess
import os
import webbrowser
import time
import shutil
from datetime import datetime

st.set_page_config(page_title="DeID Pipeline", layout="centered")


#st.markdown(md_style, unsafe_allow_html=True)

# two columns, define column widths
col_left, col_right = st.columns([3.5, 3.5], gap="large")

# Left sidebar - Labeling Section
with col_left:
    st.header("📋 Document Labeling")
    st.markdown("**Project:** Deidentifier")
    
    if st.button("🔗 Open LabelStudio"):
        try:
            # Open LabelStudio in default browser
            subprocess.Popen(["label-studio", "start"])
            # project_url = "http://localhost:8084/projects/1/data?tab=1"
            success_label = ("Opening LabelStudio in browser.\nThis may take a few seconds."
                            "\nThis may take a few seconds."
                            "\n\nPlease then import, label and export your data in the 'Deidentifier' project."
                            "\nMake sure to export as JSON-MIN."
            )

            st.success(success_label)
            #time.sleep(10)
            #webbrowser.open(f"{project_url}")
            #st.markdown(f"[➡️ Open Deidentifier Project]({project_url})", unsafe_allow_html=True)
        except:
            st.error("Could not start LabelStudio. Make sure it's installed.")


    st.divider()
    st.subheader("Import from Downloads")

    # Get Downloads folder
    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
    
    n_secs = 5
    # Auto-refresh toggle
    auto_refresh = st.checkbox(f"Auto-refresh Downloads every {n_secs} seconds", value=True)

    # List JSON files
    if os.path.exists(downloads_path):
        json_files = [f for f in os.listdir(downloads_path) if f.lower().endswith('.json')]
        json_files.sort(key=lambda x: os.path.getmtime(os.path.join(downloads_path, x)), reverse=True)
        
        if json_files:
            st.write("**Recent JSON files in Downloads:**")
            for file in json_files[:3]:   # limit to 3 most recent
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.text(file)
                with col_b:
                    if st.button("Move", key=file):
                        src = os.path.join(downloads_path, file)
                        os.makedirs("data/raw", exist_ok=True)
                        dst = os.path.join("data/raw", file)
                        
                        try:
                            shutil.move(src, dst)
                            st.success(f"➡️✅")
                            st.toast(f"Moved to data/raw/{file}")
                            time.sleep(3)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
        else:
            st.info("No JSON files found in Downloads")
    else:
        st.warning("Downloads folder not found")

    if auto_refresh:
        time.sleep(n_secs)
        st.rerun()

# Right area - Pipeline

# st.markdown(md_style, unsafe_allow_html=True)
if not auto_refresh:
    with col_right:
        st.markdown("<h2>Pipeline</h2>", unsafe_allow_html=True)
        password = st.text_input("Enter password", type="password")
        
        if st.button("🚀 Run Full Pipeline", type="primary"):
            if password == "NoFriendsAtDusk":
                with st.spinner("Running pipeline..."):
                    try:
                        st.write("Generating synthetic data...")
                        subprocess.run([
                            "python", "-m", "src.data_gen.generate_synthetic_data",
                            "--output", "data/processed/synthetic_labeled.csv",
                            "--n-samples", "600", "--sensitive-ratio", "0.5"
                        ], check=True)

                        st.write("Training model...")
                        subprocess.run(["python", "-m", "src.training.train", "--config", "configs/train.yaml"], check=True)

                        st.write("Hyperparameter tuning...")
                        subprocess.run(["python", "-m", "src.training.optuna_tune", "--config", "configs/optuna.yaml"], check=True)

                        st.write("Ranking hard datapoints...")
                        subprocess.run(["python", "-m", "src.evaluation.hard_datapoints", "--config", "configs/hard_points.yaml"], check=True)

                        st.success("Full pipeline completed!")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.error("Wrong password")