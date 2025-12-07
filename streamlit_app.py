"""
Entry point for Streamlit Cloud deployment.
This file imports and runs the actual app from ift6758/milestone_3/streamlit/
"""
import sys
from pathlib import Path

# Add the milestone_3 directory to the path so imports work correctly
milestone_3_path = Path(__file__).parent / "ift6758" / "milestone_3"
sys.path.insert(0, str(milestone_3_path))

# Execute the actual streamlit app
exec(open("ift6758/milestone_3/streamlit/streamlit_app.py").read())
