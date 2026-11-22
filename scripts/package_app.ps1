# Stage the torch-free BandScribe app and measure its size (phase 5 of the audio.cpp plan).
#
#   pwsh scripts/package_app.ps1            # -> .cache/pkg/BandScribe + BandScribe.7z
#
# Layout (desktop/main.go and bandscribe/runtime/audiocpp.py look for exactly this):
#   BandScribe.exe            MyGo desktop host
#   python/                   CPython 3.11 embeddable + site-packages from .venv-notorch
#   python/audiocpp/          audiocpp_cli.exe, cuBLAS DLLs, model_specs/
#   app/bandscribe, app/gui/dist, app/bin/bandscribe-export.exe
# Model weights are not included: they download on first use (AGENTS.md).
param(
    [string]$Out = "$PSScriptRoot\..\.cache\pkg",
    [string]$PythonVersion = "3.11.9"
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path "$PSScriptRoot\..").Path
$Out = [IO.Path]::GetFullPath($Out)
$Stage = Join-Path $Out "BandScribe"
$Venv = Join-Path $Root ".venv-notorch"
$Runtime = Join-Path $Root ".cache\audiocpp-runtime"
foreach ($p in @("$Venv\Lib\site-packages", "$Runtime\audiocpp_cli.exe", "$Root\gui\dist\index.html")) {
    if (-not (Test-Path $p)) { throw "missing $p (see AGENTS.md: .venv-notorch, native/audiocpp/build.ps1, npm run build)" }
}
if (Test-Path $Stage) {
    if (-not $Stage.StartsWith($Out)) { throw "refusing to clear $Stage" }
    Remove-Item -LiteralPath $Stage -Recurse -Force
}
New-Item -ItemType Directory -Force "$Stage\python", "$Stage\app\bin", "$Stage\python\audiocpp" | Out-Null

# Python embeddable distribution
$zip = Join-Path $Out "python-$PythonVersion-embed-amd64.zip"
if (-not (Test-Path $zip)) {
    Invoke-WebRequest "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip" -OutFile $zip
}
Expand-Archive -LiteralPath $zip -DestinationPath "$Stage\python"
Set-Content -Encoding ascii "$Stage\python\python311._pth" "python311.zip`n.`nLib\site-packages`n..\app`nimport site"

# site-packages without caches, the editable project link and venv helpers
robocopy "$Venv\Lib\site-packages" "$Stage\python\Lib\site-packages" /E /NFL /NDL /NJH /NJS /NP `
    /XD __pycache__ bandscribe-0.1.0.dist-info /XF *.pyc _editable_impl_bandscribe.pth _virtualenv.pth _virtualenv.py | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy site-packages failed ($LASTEXITCODE)" }

# audio.cpp runtime + MSVC runtime DLLs it needs (the embeddable Python ships only vcruntime)
Copy-Item "$Runtime\audiocpp_cli.exe", "$Runtime\cublas64_13.dll", "$Runtime\cublasLt64_13.dll" "$Stage\python\audiocpp"
Copy-Item -Recurse "$Runtime\model_specs" "$Stage\python\audiocpp\model_specs"
foreach ($dll in "msvcp140.dll", "vcomp140.dll", "vcruntime140.dll", "vcruntime140_1.dll") {
    Copy-Item "$env:SystemRoot\System32\$dll" "$Stage\python\audiocpp"
    if (-not (Test-Path "$Stage\python\$dll")) { Copy-Item "$env:SystemRoot\System32\$dll" "$Stage\python" }
}

# app code, GUI, Go exporter, desktop host
robocopy "$Root\bandscribe" "$Stage\app\bandscribe" /E /NFL /NDL /NJH /NJS /NP /XD __pycache__ /XF *.pyc | Out-Null
robocopy "$Root\gui\dist" "$Stage\app\gui\dist" /E /NFL /NDL /NJH /NJS /NP | Out-Null
Push-Location $Root
go build -trimpath -ldflags "-s -w" -o "$Stage\app\bin\bandscribe-export.exe" ./cmd/bandscribe-export
Pop-Location
Push-Location "$Root\desktop"
go build -trimpath -ldflags "-s -w -H=windowsgui" -o "$Stage\BandScribe.exe" .
Pop-Location

# Precompile bytecode: a cold start otherwise compiles numpy/scipy/librosa on first launch.
& "$Stage\python\python.exe" -m compileall -q -j 0 "$Stage\python\Lib\site-packages" "$Stage\app" | Out-Null

# sizes
function SizeMB($path) { [math]::Round(((Get-ChildItem $path -Recurse -File | Measure-Object Length -Sum).Sum) / 1MB, 1) }
$report = [ordered]@{
    total_mb = SizeMB $Stage
    python_mb = (SizeMB "$Stage\python") - (SizeMB "$Stage\python\audiocpp")
    audiocpp_mb = SizeMB "$Stage\python\audiocpp"
    app_mb = SizeMB "$Stage\app"
    host_mb = [math]::Round((Get-Item "$Stage\BandScribe.exe").Length / 1MB, 1)
}
$seven = "C:\Program Files\7-Zip\7z.exe"
if (Test-Path $seven) {
    $archive = Join-Path $Out "BandScribe.7z"
    if (Test-Path $archive) { Remove-Item -LiteralPath $archive }
    & $seven a -t7z -mx=9 -m0=lzma2 -mmt=on $archive "$Stage\*" | Out-Null
    $report.archive_7z_mb = [math]::Round((Get-Item $archive).Length / 1MB, 1)
}
$report | ConvertTo-Json | Tee-Object (Join-Path $Out "size.json")
