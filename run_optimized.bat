@echo off
echo Starting DrugTargetFinder with optimized settings...
set NODE_OPTIONS=--max-old-space-size=8192
streamlit run drug_target_finder.py --server.maxUploadSize=1000 --server.maxMessageSize=1000