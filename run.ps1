Push-Location "$PSScriptRoot\new-shoyuDetection"
try {
    & "$PSScriptRoot\.venv39\Scripts\python.exe" ".\MainProgram.py"
}
finally {
    Pop-Location
}
