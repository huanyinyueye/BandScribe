# Building BandScribe

These instructions describe the Windows development build. The default pipeline does not
install PyTorch. Audio.cpp model weights download and convert on first use; do not put
model weights in the installer or commit them to the repository.

## Requirements

- Windows 11 x64
- Python 3.11 and `uv`
- Node.js 24 and npm
- Go 1.27 or newer
- Visual Studio 2022 C++ build tools and CMake 3.20 or newer for native builds
- CUDA Toolkit 13.1 for the audio.cpp CUDA build

## Python environment

From the repository root:

```powershell
uv sync
```

The default environment is torch-free. To enable the optional advanced torch engine:

```powershell
uv sync --extra torch
```

Build the frontend and check the Python installation:

```powershell
Push-Location gui
npm ci
npm run build
Pop-Location
uv run bandscribe doctor
```

To run the desktop window in development, build the GUI first:

```powershell
Push-Location desktop
go run .
Pop-Location
```

The Go desktop host starts the Python service from the repository `.venv` and opens the
MyGo window. It is a development host; it does not package Python or model weights.

## Go exporter

Build the standalone score exporter from the repository root:

```powershell
go build -o .\bin\bandscribe-export.exe .\cmd\bandscribe-export
```

The desktop host and exporter use separate Go modules. Build the host from `desktop/` as
shown above.

## Native libraries

### audio.cpp CUDA runtime

Install CUDA Toolkit 13.1 and Visual Studio 2022 C++ build tools. Then run:

```powershell
pwsh .\native\audiocpp\build.ps1 -CudaArch 89
```

The script checks out the pinned audio.cpp revision, applies
`native/audiocpp/bs-roformer-multistem.patch`, builds CUDA targets for the architectures
listed in the script, and stages the CLI, model specs, and CUDA libraries in
`.cache/audiocpp-runtime/`. Set `-Cuda` to the CUDA Toolkit directory and `-CudaArch` to
the GPU architectures to compile when the defaults do not match the machine.
For example, `89` targets an RTX 40 series GPU and keeps the build focused on that
machine. Use a semicolon-separated list, such as `"86;89"`, when one build must support
multiple GPU generations.

The first run needs network access to fetch the pinned source. Model weights are not
included in this runtime.

## Torch-free desktop package

Build the GUI and audio.cpp runtime first. Create the no-torch environment and package:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-notorch"
uv sync --no-dev
Remove-Item Env:UV_PROJECT_ENVIRONMENT
pwsh .\scripts\package_app.ps1
```

The script stages embeddable Python, app files, GUI, Go exporter, MyGo host, audio.cpp,
and CUDA libraries under `.cache/pkg/BandScribe/`. It reports the unpacked size and
creates a 7-Zip archive when 7-Zip is installed. Model weights remain external and
download on first use.

## Build checks

Run the relevant commands after changing a component:

```powershell
Push-Location gui; npm run build; Pop-Location
Push-Location desktop; go build .; Pop-Location
go build .\cmd\bandscribe-export
uv run ruff check bandscribe
uv run python -m compileall -q bandscribe
```

The repository verification command is:

```powershell
uv run pytest -q
uv run ruff check bandscribe
Push-Location gui; npm run build; Pop-Location
```

The test suite includes GPU and slow tests; use the markers in `pyproject.toml` when the
machine cannot run those checks. A successful build alone does not verify a complete
transcription. Record real end-to-end results in `docs/e2e/<date>-<slug>/result.md`.
