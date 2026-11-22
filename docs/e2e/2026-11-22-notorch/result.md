# 無 torch 流程：接上 BandScribe、整首 E2E、打包（2026-11-22，第 3–5 階段）

## 接上的東西

- 分軌 `bs_roformer_sw` 預設走 audio.cpp（`bandscribe/separation/audiocpp_separator.py`），MuScriptor 走 audio.cpp request sequence，一次載入跑完所有 job（`AudiocppMuScriptorTranscriber`），拍點走 ONNX Runtime DirectML。`BANDSCRIBE_ENGINE=torch` 可切回 torch。
- 第一次使用時在本機轉檔，不用 torch：SW ckpt → safetensors（和 torch 轉換器輸出的 1915 個張量逐一相同，約 2 s），Beat This! 重建 ONNX 權重，MuScriptor 直接讀 Hugging Face snapshot（`bandscribe/models/prepare.py`）。
- 顯卡偵測改用 `audiocpp_cli --list-devices --json`；`doctor` 多了 engine 檢查，沒裝 torch 時 torch 檢查顯示為選用。
- GPU 失敗會自動退回 CPU：audio.cpp 單次呼叫 CUDA 失敗就用 CPU 重跑；DirectML 當掉（這次實際遇到 DXGI 887A0006）就重開 CPU session。
- `pyproject.toml`：torch、beat-this、bs-roformer-infer、demucs、muscriptor 等移到 `torch` extra；預設安裝改成 `onnxruntime-directml`。

## 整首 E2E（沒有 torch 的乾淨環境）

`.venv-notorch`（`UV_PROJECT_ENVIRONMENT=.venv-notorch uv sync --no-dev`，557 MB；目前 torch 環境 4,959 MB）。輸入 `09 Neo-Aspect.wav`（310 s），設定同 `out/neo-guitar-pc`：balanced、只轉吉他、Drop D、雙模型共識與 ±1 半音集成照舊。從匯入跑到匯出 `.gp/.gp5/.mid/.musicxml` 全部成功，程序內 `torch` 從未載入。輸出 `out/neo-guitar-cpp`。

對人工 GP（Distortion Guitar 3542 音，offset 0.172125，`--map "guitar=Distortion Guitar"`）：

| | torch（out/neo-guitar-pc） | 無 torch（audio.cpp + ONNX） |
|---|---|---|
| 最終吉他 note F1 | 0.671（P 0.611 / R 0.743） | **0.675**（P 0.607 / R 0.762） |
| MuScriptor 原始 F1 | 0.681 | 0.691 |
| Basic Pitch 原始 F1 | 0.213 | 0.213 |
| onset F1 | 0.768 | 0.762 |

時間：這次無法公平比較。整段執行期間顯卡同時被 `D:\factory\UTAU` 的分軌程式（55–94% 使用率）、ComfyUI 和另一個 BandScribe 測試佔用，MuScriptor 程序有時只分到 3% GPU。實測：拍點 92 s（DirectML 當掉後改 CPU）、分軌 117.7 s、MuScriptor 三個 310 s job 共 3441 s、Basic Pitch 27 s，總計 3866 s。同一段 60 s 在乾淨顯卡約 10 s，這次搶卡時要 135 s。乾淨狀態的數字見 `docs/e2e/2026-11-22-audiocpp-gpu/result.md`（分軌 117.6 s 對 torch 243 s）。

## 打包

`scripts/package_app.ps1`：CPython 3.11.9 embeddable + `.venv-notorch` 的 site-packages（含預編譯 .pyc）、audio.cpp CLI + cuBLAS + MSVC runtime、`app/`（bandscribe、gui/dist、Go 匯出器）、MyGo 桌面殼。模型不打包。

| | MB |
|---|---|
| 解壓後總計 | 1290.8 |
| Python + 套件 | 735.4 |
| audio.cpp + cuBLAS | 532.8 |
| app + 桌面殼 | 22.6 |
| **7z LZMA2 壓縮後（約等於安裝檔）** | **553.3** |

打包後直接執行 `BandScribe.exe`：7 s 內後端就緒，視窗開出匯入頁（`packaged-app.png`），關窗後後端正常關閉。打包版 Python 找得到內附 audio.cpp、CUDA（RTX 4070）和 DirectML，沒有 torch。

## 沒驗證

- 乾淨顯卡下的整首總時間，以及同一台機器的 torch 整首時間。
- 打包版從匯入跑完整首轉譜（只驗證啟動、引擎偵測、視窗）；沒裝 CUDA Toolkit / 沒有 NVIDIA 卡的電腦。
- 真正的安裝程式（Inno/NSIS）；7z 大小是估計值。
- 段落偵測這次失敗：另一個工作階段正在建 `native/bsdsp`，`bin/bsdsp.dll` 被鎖住，和 audio.cpp 無關。
