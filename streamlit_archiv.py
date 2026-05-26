# Removed from above
    '''
    uploaded_files = st.file_uploader(
        "Upload documents (JSON)", # could add csv or txt in future
        accept_multiple_files=True,
        type=['json'] # in future could be csv or txt
    )
    '''
    ''' Not implemented at the moment. Just direct people to Labelstudio
    if uploaded_files and st.button("Import to LabelStudio"):
        os.makedirs("temp_upload", exist_ok=True)
        saved_files = []
        
        for file in uploaded_files:
            path = f"temp_upload/{file.name}"
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            saved_files.append(path)
        
        st.success(f"✅ Files saved to temp_upload/")
        st.info("→ Go to your LabelStudio project → Click **Import** → Select these files from the temp_upload folder")
        st.markdown("**Files ready:** " + ", ".join([f.name for f in uploaded_files]))
    '''

'''
subprocess.run([
                "label-studio", "export", 
                "Deidentifier",          # Try project name first
                "JSON_MIN", 
                f"--export-path={output_path}"
            ], check=True, capture_output=True, text=True)
'''

'''with col_left:
    st.header("📋 Document Labeling")
    st.markdown("**Project:** Deidentifier")

    
    if st.button("🔗 Open LabelStudio"):
        try:
            # Open LabelStudio in default browser
            subprocess.Popen(["label-studio", "start"])
            project_url = "http://localhost:8084/projects/1/data?tab=1"
            st.success("Opening LabelStudio in browser.\nThis might take a few seconds.\n\nPlease then import, label and export your data in the 'Deidentifier' project.")
            #time.sleep(10)
            #webbrowser.open(f"{project_url}")
            #st.markdown(f"[➡️ Open Deidentifier Project]({project_url})", unsafe_allow_html=True)
        except:
            st.error("Could not start LabelStudio. Make sure it's installed.")

    st.divider()
    #if st.button("⬇️ Export Labeled Data (JSON-MIN)"):
    if st.button("⬇️ Move Labeled Data from Downloads to project data file."):
        try:
            #output_path = "data/raw/labeled_from_ls.json"
            st.info("Export can take a few seconds, thank you for your patience.")
            output_path = "downloads/labeled_from_ls.json"
            
            
            
            if os.path.exists(output_path):
                st.success(f"✅ Exported successfully to:\n{os.path.abspath(output_path)}")
            else:
                st.error(f"Export command ran to:\n{os.path.abspath(output_path)}\n\nbut file not found.")
                
        except subprocess.CalledProcessError as e:
            st.error(f"Export failed: {e.stderr if e.stderr else e}")
            st.info("Tip: Try using the project **ID** instead of 'Deidentifier'")
'''