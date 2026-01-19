$TargetFile = "$PSScriptRoot\start_5090_services.bat"
$ShortcutFile = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\Start5090Services.lnk"
$WScriptShell = New-Object -ComObject WScript.Shell
$Shortcut = $WScriptShell.CreateShortcut($ShortcutFile)
$Shortcut.TargetPath = $TargetFile
$Shortcut.WorkingDirectory = $PSScriptRoot
$Shortcut.Save()

Write-Host "Shortcut created at: $ShortcutFile"
Write-Host "Target: $TargetFile"
