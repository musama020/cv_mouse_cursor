' Silent background launcher — runs the combined app (gaze + scroll) in the tray.
' To run only one feature, change "both" below to "gaze" or "scroll".
Set WshShell = CreateObject("WScript.Shell")
folder = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
WshShell.Run "pythonw """ & folder & "\handler.py"" both", 0, False
