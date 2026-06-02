Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "pythonw """ & CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName) & "\gazeswitch.py""", 0, False
