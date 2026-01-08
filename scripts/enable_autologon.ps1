# Enable Windows Auto-Login via Registry
# Usage: Run as Administrator
param (
    [string]$Username = "kenny",
    [string]$Password,
    [string]$Domain = $env:COMPUTERNAME
)

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Warning "You do not have Administrator rights to run this script!`nPlease re-run this script as an Administrator!"
    break
}

if ([string]::IsNullOrEmpty($Password)) {
    $Password = Read-Host "Please enter the password for user '$Username'" -AsSecureString
    $Password = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto([System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($Password))
}

$RegistryPath = "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon"

Write-Host "Configuring AutoLogon for user '$Username'..."

try {
    Set-ItemProperty -Path $RegistryPath -Name "DefaultUserName" -Value $Username -Force
    Set-ItemProperty -Path $RegistryPath -Name "DefaultPassword" -Value $Password -Force
    Set-ItemProperty -Path $RegistryPath -Name "DefaultDomainName" -Value $Domain -Force
    Set-ItemProperty -Path $RegistryPath -Name "AutoAdminLogon" -Value "1" -Force
    
    # Optional: Disable 'Require Hello Sign-in' which can block auto-login on some builds
    $PasswordlessPath = "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\PasswordLess\Device"
    if (Test-Path $PasswordlessPath) {
        Set-ItemProperty -Path $PasswordlessPath -Name "DevicePasswordLessBuildVersion" -Value 0 -Type DWord -Force
    }
    
    Write-Host "[SUCCESS] Auto-Login configured successfully!" -ForegroundColor Green
    Write-Host "Please reboot the device to test." -ForegroundColor Yellow
}
catch {
    Write-Error "Failed to set registry keys: $_"
}
