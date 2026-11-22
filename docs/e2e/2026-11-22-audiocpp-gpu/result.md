# audio.cpp CUDA MuScriptor vs torch（2026-11-22）

輸入：`out/neo-guitar-pc/stems/guitar.wav` 第 30–90 秒（60 秒，44.1 kHz），吉他三種 instrument 限制，greedy 解碼。RTX 4070，CUDA 13.1。

| | torch 2.11 cu128 | audio.cpp CUDA（sm_89，只編 muscriptor/roformer/demucs） |
|---|---:|---:|
| 時間 | 載入 9.7 s + 轉譜 94.0 s（不含 Python import） | 整個程序 50.7 s（含載入） |
| 音符數 | 958 | 1008 |
| 對人工 GP（Distortion Guitar，762 音，50 ms，offset 0.172125，延遲修正 30 ms） | P 0.572 / R 0.719 / **F 0.637** | P 0.575 / R 0.761 / **F 0.655** |
| 執行環境體積 | `torch` 4.08 GiB | `audiocpp_cli.exe` 54 MiB + `cublasLt64_13` 428 + `cublas64_13` 49 + `cudart64_13` 1 MiB ≈ 532 MiB |

兩者互相比對（同音高、起音 50 ms 內）F1 0.914。

## 分軌：audio.cpp HTDemucs 6-stem

整首 `09 Neo-Aspect.wav`（310 s）用 audio.cpp CUDA HTDemucs 6-stem（F16 GGUF 52 MiB）分軌，整個程序 55.9 s。取其 guitar 第 30–90 秒交給 audio.cpp MuScriptor：1625 音，P 0.281 / R 0.598 / **F 0.382**，遠低於 BS-RoFormer SW 吉他軌的 0.655，不能取代。

## 分軌：audio.cpp BS-RoFormer SW 6-stem（第 1 階段）

audio.cpp 原本只支援單軌 BS-RoFormer。修改在 `native/audiocpp/bs-roformer-multistem.patch`（基於 commit 04a37f2）：每軌各載一組 `mask_estimators.<n>`，圖輸出疊成 `{軌數, 幀, 維度}`，每軌各做遮罩、ISTFT、overlap-add，輸出檔名照 config 的 `instruments`。轉換器補上 SW 設定缺的 `instruments`、預設 band 切法與取樣率。重建腳本：`native/audiocpp/build.ps1`。

整首 `09 Neo-Aspect.wav`（310 s）：

| | torch（BandScribe 現行，FP16 autocast，batch 2） | audio.cpp CUDA（safetensors F32） |
|---|---:|---:|
| 分軌時間 | 242.95 s（manifest） | 運算 92.8 s，整個程序 117.6 s（含載入） |
| 記憶體 | 顯示記憶體峰值 3.97 GB | RAM 峰值 2.65 GB |

逐軌和 torch 輸出比較（不計增益；torch 版輸入先做了 −1 dBFS 峰值正規化，所以有固定 0.8936 倍）：

| 軌 | SI-SDR（對 torch） |
|---|---:|
| bass | 58.9 dB |
| drums | 58.8 dB |
| other | 52.7 dB |
| vocals | 59.7 dB |
| guitar | 56.3 dB |
| piano | 53.6 dB |

吉他軌第 30–90 秒交給 audio.cpp MuScriptor：1017 音，P 0.575 / R 0.768 / **F 0.658**，達到門檻 0.655（torch 全程 0.637）。

權重：Hugging Face 快取的 `MuScriptor/muscriptor-medium` 以 audio.cpp `tests/muscriptor/convert_gguf.py` 轉為 F32 GGUF（1.14 GiB），不打包，安裝後下載／轉換。

未驗證：整首歌、其他歌曲、雙模型共識後的 F1、BS-RoFormer／HTDemucs 的 audio.cpp 版本（分軌目前仍需 torch）、顯示記憶體用量、未裝 CUDA Toolkit 的電腦上只附 DLL 能否執行。

腳本：`.cache/bench-audiocpp/`（torch_bench.py、compare.py、score_ref.py）。

## 拍點：Beat This! → ONNX（第 2 階段）

`scripts/export_beat_this_onnx.py` 把 `final0.ckpt` 匯出成 ONNX 圖（2.3 MB，權重放在外部檔），另寫一份 manifest，記錄每個權重在 checkpoint 的鍵與位元組範圍。App 下載 `final0.ckpt` 後，`bandscribe.models.ckpt_reader` 不用 torch 讀出權重，`beat_onnx.build_weights` 重建 77 MiB 權重檔，sha256 和匯出時相同（實測逐位元相同，5.3 s）。這個讀取器也對 SW checkpoint 驗證過：1939 個張量和 `torch.load` 完全相同。

推論在 `bandscribe/analysis/beat_onnx.py`：numpy log-mel（和 torchaudio 差 8e-5）、1500 幀分塊、upstream 的 minimal 峰值挑選。GPU 走 ONNX Runtime DirectML（`DirectML.dll` 17 MiB，任何 DX12 顯卡；CUDA 版 ORT 要 cuDNN，會超過 1 GB 預算）。

整首 Neo-Aspect：

| | torch CUDA FP32 | torch CUDA FP16（現行） | ONNX DirectML |
|---|---:|---:|---:|
| 拍點 | 654 | 654 | 654（和 FP32 完全相同） |
| 強拍 | 172 | 173 | 172（和 FP32 完全相同） |
| 小節 | 172 | 173 | 172（起點、拍號和 FP32 完全相同） |
| 時間 | — | 首次 32.7 s（含 CUDA 初始化），熱機 2.0 s | 乾淨程序（無 torch）載入 2.5 s + 追蹤 2.6 s |

FP16 多出的那個強拍在 26.02 s，logit 是 +0.003（FP32 是 −0.007），只是門檻 0 附近的捨入差。CPU 版 ONNX 結果相同，但要約 21 s。