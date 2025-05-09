@echo off
REM Check if the virtual environment exists; if not, create it.
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
pip install --upgrade pip

echo Installing project requirements...
pip install -r requirements.txt

echo All requirements installed successfully!
pause
