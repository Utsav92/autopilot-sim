# Run in an ELEVATED PowerShell (installing WSL needs admin and may need a reboot).
#   powershell -ExecutionPolicy Bypass -File bridge\setup_wsl.ps1
# This changes system configuration (WSL feature + Ubuntu distro) and downloads ~15 GB, so it is never run automatically.
$ErrorActionPreference = 'Stop'
$distros = (wsl.exe -l -q 2>$null) -replace "`0", '' | Where-Object { $_.Trim() }
if (-not $distros) {
  Write-Host 'No WSL distro found. Installing Ubuntu-24.04 (a reboot may be required; re-run this script afterwards).'
  wsl.exe --install -d Ubuntu-24.04
  exit 0
}
$here = (Resolve-Path "$PSScriptRoot").Path
$wslPath = (wsl.exe wslpath -a ($here -replace '\\', '/')).Trim()
Write-Host "Using bridge scripts from $wslPath"
wsl.exe bash -lc "bash '$wslPath/wsl_setup.sh'"
Write-Host 'Setup finished. Start the sim with:  wsl bash -lc "bash ''' + $wslPath + '/run_sim.sh''"'
Write-Host 'Then open the AutoPilot Sim page and press "Connect" in the openpilot live panel.'
