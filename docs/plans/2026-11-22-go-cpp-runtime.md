# 執行環境從 Python 換成 Go + C++（計畫）

狀態：草案，等 grilling 收斂後才動工。授權與分工已定（2026-11-22）。

## 目標

安裝包與執行時不依賴 Python。量測（2026-11-22）：`.venv` 4.76 GB，其中 `torch` 4.18 GB、
其餘套件 661 MB、Python 3.11 直譯器 85 MB。模型推論換成 audio.cpp 由另一個 session 負責；
本計畫負責其餘部分（約 750 MB 加上 Node 與 alphaTab）。

不是目標：加速。Python 本身的處理一首歌不到 8 秒（`out/singalive/manifest.json`）。

## 已定決策

- 專案授權 Apache-2.0（`LICENSE`）。
- Go：外層流程、設定、音訊讀寫、符號處理（共識、量化、TAB、QA、版面）、所有匯出、GUI 伺服器。
- C++：DSP，編成 C ABI 動態函式庫 `bsdsp.dll`，Go 以 cgo 呼叫。與 audio.cpp 共用 MSVC / CMake 工具鏈。
- 模型推論：audio.cpp（另一個 session）。還沒移植完的模型暫時由 Python 模型程序執行。
- `.gp`：把 alphaTab 1.9.0（MPL-2.0）實際用到的匯出路徑翻成 Go；這些檔案保留 MPL-2.0 檔頭，
  其餘程式碼不受影響。alphaTab 只留在測試當標準答案，不打包。
- `.gp5`：不參考 pyguitarpro（LGPL）程式碼，照格式自己寫；測試用 pyguitarpro 讀回驗證。
- rubberband（GPL-2.0-or-later）三層：只當獨立執行檔、只交換 WAV；選用元件，放
  `third_party/rubberband/` 附 COPYING 與原始碼連結，缺了照常運作；Roselia 吉他 F1 贏過
  librosa 對齊版才收。

## 元件對照

| 現在（Python） | 之後 | 授權 |
|---|---|---|
| typer、pyyaml | Go `flag`、`gopkg.in/yaml.v3` | BSD / MIT·Apache |
| fastapi、uvicorn、pydantic、pywebview | `net/http` + `github.com/coder/websocket`；視窗用 MyGo | BSD / ISC / — |
| soundfile、imageio-ffmpeg | `github.com/mewkiz/flac`、`github.com/hajimehoshi/go-mp3`、`github.com/go-audio/wav` | Unlicense / Apache / Apache |
| soxr（VHQ 重取樣） | `bsdsp`，動態連結 libsoxr | LGPL-2.1（動態連結） |
| librosa `pitch_shift`、`melspectrogram`、`chroma_stft`、`chroma_cqt` | `bsdsp`，FFT 用 pocketfft | BSD-3 |
| numpy（符號處理部分）、`fractions` | Go `math/big.Rat` | BSD |
| mido、MusicXML（自寫） | `gitlab.com/gomidi/midi/v2`、`encoding/xml` | MIT / BSD |
| pyguitarpro（`.gp5`） | Go 自寫 | — |
| Node + alphaTab（`.gp`） | Go 翻譯（MPL-2.0 檔案） | MPL-2.0 |
| huggingface_hub | Go HTTP 下載 + sha256 | — |
| onnxruntime（Basic Pitch、ADTOF） | audio.cpp，或 `github.com/yalue/onnxruntime_go` | MIT |
| mir_eval | 只留在 `benchmarks/`、`scripts/`，不打包 | — |

## 正確性標準

- 決定性的部分（符號處理、匯出）：同一份輸入，Go 輸出與 Python / alphaTab 一致。`.gp` 解開 zip
  後每個檔案正規化比對完全相同；`score.json`、MIDI、MusicXML 完全相同。
- DSP：逐項對 librosa / soxr，float32 誤差上限另在 Phase 2 量測後訂；最終以 Roselia 吉他 F1 不退步為準。
- 參考資料：先用 Python 對固定歌曲傾印每一階段的中間結果，當 Go / C++ 的標準答案。

## 階段（每階段結束都有一次真實 E2E）

0. **基準傾印**：Python 傾印每階段中間結果到 `.cache/parity/<song>/`；加極端案例產生器
   （所有時值、附點、連音、跨小節延音、每拍速度、空小節、鼓、4–8 弦、F 譜號、大小調、段落、Unicode）。
1. **匯出改 Go**：MIDI、MusicXML、`.gp5`、`.gp`。Python 改呼叫 Go 執行檔匯出，Node 與 pyguitarpro
   退出執行環境。E2E：整首歌匯出，GP8 打開參考譜面。
2. **`bsdsp`（C++）**：重取樣、STFT/iSTFT、phase vocoder、`pitch_shift`、mel、chroma、CQT。
   Python 改呼叫 `bsdsp`，librosa、numba、llvmlite 退出。E2E：Roselia 吉他 F1。rubberband 在這階段評估。
3. **符號處理改 Go**：共識、投票、鼓、強力和弦、速度 / 拍號 / 調性 / 段落、量化、TAB DP、QA、版面。
   E2E：同樣的模型輸出，`score.json` 與匯出和 Python 版一致。
4. **流程改 Go**：階段排程、`manifest.json`、快取、進度與取消、設定、模型註冊與下載、CLI。
   模型透過介面呼叫：audio.cpp 或暫時的 Python 模型程序（stdio JSON + WAV）。E2E：整首歌從頭跑到尾。
5. **GUI 伺服器改 Go**：FastAPI 端點與 websocket 移到 Go，MyGo 視窗在同一個程序內提供。
   E2E：桌面視窗實際操作截圖。
6. **拔掉 Python**：所有模型都在 audio.cpp 後，刪除 Python 模型程序；量測安裝包大小。

## 風險

- `pitch_shift` 的輸出直接進 MuScriptor，數值差異會影響 F1。
- `chroma_cqt` 只有 GUI 快速分析在用，移植成本最高、價值最低。
- `.gp` 必須涵蓋 `score.finish()` 對我們會用到的模型所做的正規化。
- 另一個 session 有未提交的改動（`desktop/`、`gui/`、`bandscribe/gui/server.py`、`README.md`、
  `docs/INTERFACES.md`）；它提交前本計畫不碰這些檔案。
