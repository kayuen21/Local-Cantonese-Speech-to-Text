' Cantonese Speech to Text - double-click this file to put a shortcut on the desktop.
Option Explicit

Dim fso, shell, here, shortcut
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

here = fso.GetParentFolderName(WScript.ScriptFullName)
Set shortcut = shell.CreateShortcut(shell.SpecialFolders("Desktop") & "\Cantonese Speech to Text.lnk")
shortcut.TargetPath = here & "\CantoneseSTT.exe"
shortcut.WorkingDirectory = here
shortcut.Description = "Local Cantonese speech to text"
If fso.FileExists(here & "\app.ico") Then
  shortcut.IconLocation = here & "\app.ico"
End If
shortcut.Save

MsgBox "Done - look for ""Cantonese Speech to Text"" on your desktop.", 64, "Cantonese Speech to Text"
