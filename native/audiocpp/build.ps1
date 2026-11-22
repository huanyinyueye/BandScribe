# Builds the audio.cpp CLI that BandScribe uses for separation (BS-RoFormer SW)
# and transcription (MuScriptor), then stages a self-contained runtime folder.
#   pwsh native/audiocpp/build.ps1 [-Src <dir>] [-Out <dir>] [-CudaArch 89]
param(
    [string]$Src = "$env:LOCALAPPDATA\bandscribe\audiocpp-src",
    [string]$Out = "$PSScriptRoot\..\..\.cache\audiocpp-runtime",
    [string]$CudaArch = "75;86;89;120",
    [string]$Cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
)
$ErrorActionPreference = "Stop"
$commit = "04a37f2e80ff2dbcdd84cc67848a970bd02a73e0"
$patch = Join-Path $PSScriptRoot "bs-roformer-multistem.patch"

if (-not (Test-Path $Src)) {
    git clone https://github.com/0xShug0/audio.cpp.git $Src
}
git -C $Src fetch --quiet origin $commit
git -C $Src checkout --quiet --force $commit
git -C $Src apply $patch

$build = Join-Path $Src "build\bandscribe-cuda"
cmake -S $Src -B $build -G "Visual Studio 17 2022" -A x64 `
    -DAUDIOCPP_MODEL_SET=custom "-DAUDIOCPP_MODELS=muscriptor;roformer" `
    -DENGINE_ENABLE_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch"
cmake --build $build --config Release --target audiocpp_cli audiocpp_gguf --parallel

New-Item -ItemType Directory -Force "$Out\model_specs" | Out-Null
Copy-Item "$build\bin\Release\audiocpp_cli.exe", "$build\bin\Release\audiocpp_gguf.exe" $Out
Copy-Item "$Cuda\bin\x64\cublas64_13.dll", "$Cuda\bin\x64\cublasLt64_13.dll" $Out
Copy-Item "$Src\model_specs\bs_roformer.json", "$Src\model_specs\muscriptor.json" "$Out\model_specs"
Write-Host "audio.cpp runtime staged in $Out"
