' run_app.vbs
Set objShell = CreateObject("WScript.Shell")
objShell.Run "cmd.exe /c streamlit run app.py", 0, True
Set objShell = Nothing
