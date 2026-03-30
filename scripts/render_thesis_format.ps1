param(
    [string]$Config = "C:\GitHub\YOLO-CV\essay\docs\thesis_format.json",
    [string]$Template = "C:\GitHub\YOLO-CV\essay\docs\thesis_format.template.tex",
    [string]$Output = "C:\GitHub\YOLO-CV\essay\docs\thesis_format.tex"
)

& "C:\GitHub\YOLO-CV\.venv\Scripts\python.exe" "C:\GitHub\YOLO-CV\scripts\render_thesis_format.py" --config $Config --template $Template --output $Output
