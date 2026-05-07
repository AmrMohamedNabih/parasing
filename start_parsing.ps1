# This script launches the three parsing services in separate PowerShell windows.
# It automatically activates the virtual environment and sets the correct directory.

$ParsingDir = Join-Path $PSScriptRoot "parsing"

function Start-ServiceWindow {
    param (
        [string]$Title,
        [string]$Command
    )
    
    # Construct the full command string
    # 1. Change directory to parsing
    # 2. Activate venv
    # 3. Set the window title
    # 4. Run the service command
    $FullCommand = "cd '$ParsingDir'; .\venv\Scripts\Activate.ps1; `$Host.UI.RawUI.WindowTitle = '$Title'; Write-Host '>>> Starting $Title' -ForegroundColor Cyan; $Command"
    
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "$FullCommand"
}

Write-Host "Launching Parsing Services..." -ForegroundColor Green

# 1. RAG Service (Port 8001)
Start-ServiceWindow -Title "RAG Service (8001)" -Command "uvicorn rag_service.main:app --reload --port 8001"

# 2. Embedding Worker
Start-ServiceWindow -Title "Embedding Worker" -Command "python embedding_worker.py"

# 3. Parsing Main API (Port 8000)
Start-ServiceWindow -Title "Parsing API (8000)" -Command "uvicorn app.main:app --reload --port 8000"
