import streamlit as st
import subprocess
import os
import webbrowser
import time
import shutil
from datetime import datetime
import sys
import pandas as pd
import subprocess
                                    
st.set_page_config(page_title="DeID Pipeline", layout="wide")

# Reduce flicker during rerun
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background-color: #0e1117 !important;
        }
        [data-testid="stHeader"] {
            background-color: #0e1117 !important;
        }
        .stColumn {
            transition: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# two columns, define column widths
col_left, col_right = st.columns([5, 5], gap="large")

# Left sidebar - Labeling Section
with col_left:
    st.header("📋 Document Labeling")
    st.markdown("**Project:** Deidentifier")
    
    if st.button("🔗 Open LabelStudio"):
        try:
            # Open LabelStudio in default browser
            subprocess.Popen(["label-studio", "start"])
            # project_url = "http://localhost:8084/projects/1/data?tab=1"
            success_label = ("Opening LabelStudio in browser."
                            "\nThis may take a few seconds."
                            "\n\nPlease then import, label and export your data in the 'Deidentifier' project."
                            "\nMake sure to export as csv."
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
    
    n_secs = 10
    # Auto-refresh toggle
    auto_refresh = st.checkbox(f"Auto-refresh Downloads every {n_secs} seconds", value=True)

    # List csv files
    if os.path.exists(downloads_path):
        #csv_files = [f for f in os.listdir(downloads_path) if f.lower().endswith('.csv')]
        csv_files = [
            f for f in os.listdir(downloads_path) 
            if f.lower().endswith('.csv') and 'project' in f.lower()
        ]
        csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(downloads_path, x)), reverse=True)
        
        if len(csv_files)>0:
            st.write("**Recent csv files in Downloads:**")
            for file in csv_files[:3]:   # limit to 3 most recent
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.text(file)
                with col_b:
                    if st.button("Process & Move", key=f"proc_{file}"):
                        src = os.path.join(downloads_path, file)
                        os.makedirs("data/human_labelled", exist_ok=True)
                        dst = os.path.join("data/human_labelled", file)
                    
                        try:
                            import pandas as pd
                            df = pd.read_csv(src)
                        
                            # Restructure to required columns
                            if 'text' in df.columns and 'label' in df.columns:
                                df = df[['text', 'label']].copy()
                                df['id'] = range(len(df))          # Add id column
                                df = df[['id', 'text', 'label']]   # Reorder
                                
                                # Remove empty rows
                                df = df.dropna(subset=['text'])
                                df = df[df['text'].astype(str).str.strip() != ""]
                                
                                df.to_csv(dst, index=False)
                                st.success("Moved ✅")
                                st.toast(f"✅ Processed and moved: {file}", icon="✅")
                                time.sleep(2)
                                #st.rerun()

                                # Push to github
                                try:
                                    with st.spinner("Pushing to GitHub..."):
                                        # Add folder
                                        subprocess.run(["git", "add", "data/human_labelled"], check=True)
                                        
                                        # Commit
                                        result = subprocess.run(["git", "commit", "-m", "Add/Update human labeled data"], 
                                                            capture_output=True, text=True)
                                        
                                        # Push
                                        subprocess.run(["git", "push"], check=True)
                                        
                                        st.success("✅ Pushed")
                                        st.toast("✅ Successfully pushed human_labelled data to GitHub")
                                        time.sleep(n_secs)
                                except subprocess.CalledProcessError as e:
                                    if "nothing to commit" in str(e.stdout) + str(e.stderr):
                                        st.info("No new changes to commit")
                                        time.sleep(n_secs)
                                    else:
                                        st.error(f"Git error: {e.stderr if e.stderr else e}")
                                        time.sleep(n_secs)
                                except Exception as e:
                                    st.error(f"Failed: {e}")
                                time.sleep(n_secs)
                            else:
                                st.error("CSV must contain at least 'text' and 'label' columns")
                        except Exception as e:
                            st.error(f"Error: {e}")
        else:
            st.info("No CSV files found in Downloads")
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
                        python_exe = sys.executable

                        st.write("Generating synthetic data...")
                        n_synth_data = "600"
                        subprocess.run([
                            python_exe, "-m", "src.data_gen.generate_synthetic_data",
                            "--output", "data/processed/synthetic_labeled.csv",
                            "--n-samples", n_synth_data, "--sensitive-ratio", "0.5"], 
                            check=True, cwd = os.getcwd())

                        st.write("Training model...")
                        subprocess.run([python_exe, "-m", "src.training.train", "--config", "configs/train.yaml"], 
                                       check=True, cwd = os.getcwd())

                        st.write("Hyperparameter tuning...")
                        subprocess.run([python_exe, "-m", "src.training.optuna_tune", "--config", "configs/optuna.yaml"], 
                                       check=True, cwd = os.getcwd())

                        st.write("Ranking hard datapoints...")
                        subprocess.run([python_exe, "-m", "src.evaluation.hard_datapoints", "--config", "configs/hard_points.yaml"], 
                                       check=True, cwd = os.getcwd())

                        st.success("Full pipeline completed!")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.error("Wrong password")
