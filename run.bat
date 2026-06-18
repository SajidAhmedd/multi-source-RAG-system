@echo off
echo Starting Multi-Source RAG Application...
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt -q

REM Run the app
echo.
echo Starting Streamlit app...
streamlit run app.py
