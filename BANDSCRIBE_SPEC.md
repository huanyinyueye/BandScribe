# BandScribe — 技術研究報告與實作規格（Opus → Fable 移交文件）

- 研究截止日：2026-11-12
- 研究者：Claude Opus 5.5（Research / Architecture）
- 實作者：Claude Fable 5.2（獨立 Agent Session）
- 文件狀態：v0.2。使用者原始需求在「3.1 音源分離模型」之後中斷；第 3.2 節以後由研究者依第 1、2 節推導。2026-11-12 使用者補充需求已併入（見第 12 節）：
  1. 必須輸出 **Guitar Pro 8 可開啟的 `.gp` 檔**
  2. GPU：**NVIDIA GeForce RTX 3080，10 GB VRAM**
  3. **僅個人使用**（非商業授權權重可用）
  4. **需要圖形介面**

---

## 0. 驗證狀態定義

| 標記 | 意義 |
|---|---|
| **V-SRC** | 已在一手來源（官方 repo / release / 論文 / model card）看到該數值或事實 |
| **V-2ND** | 僅見於二手來源（新聞、部落格、第三方 model card、搜尋摘要），未取得一手原文 |
| **CONFLICT** | 來源之間數值或日期互相矛盾 |
| **UNVERIFIED** | 研究者記憶或推論，本次未查證 |
| **MUST-BENCH** | Fable 必須在本機實測後才能作為決策依據 |

**重要限制**：本次研究透過網頁搜尋摘要取得資訊，未逐一下載權重、未執行任何本機推論。所有 SDR / F1 數字皆為「作者或平台自報」，不同評測集（MUSDB18-HQ test、MVSep Multisong、MoisesDB）之間**不可直接比較**。任何「SOTA」結論都必須經過第 6 節的本機 benchmark 才能成立。

---

## 1. 研究結論摘要（Executive Summary）

1. **沒有單一模型能可靠完成全部分軌。** 最佳策略是「通用多軌模型 → 專用單樂器模型 → 交叉驗證」的分層架構。
2. **吉他分離仍是最弱環節**：主流 6 軌模型的吉他 SDR 約 9 dB、鋼琴約 8 dB，明顯低於鼓 / 貝斯（14 dB 上下）。因此吉他轉譜必須搭配信心分數與低信心標記，**不得**假設吉他 stem 是乾淨的。
3. **BS-RoFormer SW（6 stem）是目前最合理的主幹分離器**；HTDemucs 6s 只作 baseline（其 other stem 品質明顯退化）。
4. **SubRoformer（arXiv 2611.08130，2026-11）是最新候選**，論文自報 MUSDB18-HQ 4-stem 平均 11.59 dB（含 500 首額外資料），但 MSST 收錄權重表列約 10.42 dB，數字衝突，且僅 4 stem（無吉他軌），**只能作 drums/bass/vocals 的交叉驗證候選**。
5. **轉譜方面，MuScriptor（Kyutai × Mirelo，2026-07）是目前最強的開放權重多樂器 AMT**，且已支援吉他 TAB 輸出；但**權重為 CC BY-NC 4.0（非商業）**。BandScribe 定位為本地個人工具則可用，若未來商業化必須替換。
6. **TAB 指板指派**有 2026 年的新研究（Noise2Fret、TART），但程式碼 / 權重釋出狀態未確認；實作上先以「規則 + 動態規劃（playability cost）」為保底，研究模型列為可插拔選項。

---

## 2. 音源分離模型調查（使用者需求 3.1）

### 2.1 比較表

| 模型 | Stems | 自報分數 | 評測集 | 權重 / 版本 | 授權 | 狀態 |
|---|---|---|---|---|---|---|
| **BS-RoFormer SW** | vocals, drums, bass, guitar, piano, other | vocals 11.30 / inst 17.50 / bass 14.62 / drums 14.11 / **guitar 9.05** / **piano 7.83** / other 8.71 | MVSep Multisong | checkpoint 作者 jarredou；PyPI `bs-roformer-infer`；HF ONNX 版 `Acapellas/bs-roformer-sw-6stem-FP16-FP32-onnx` | 未確認 | V-2ND（MVSep 頁面）、MUST-BENCH |
| **SCNet XL IHF** | 4 stem（vocals, drums, bass, other） | MUSDB avg 10.08（bass 9.23 / drums 11.81 / vocals 11.42 / other 7.88）；Multisong avg 9.92 | MUSDB18-HQ test | MSST release v1.0.15，`model_scnet_ep_36_sdr_10.0891.ckpt`；僅以 MUSDB18 訓練 | MSST 程式 MIT；權重未確認 | V-SRC（MSST pretrained_models.md） |
| SCNet XL（非 IHF） | 4 stem | MUSDB avg 9.80 | MUSDB18-HQ | MSST | 同上 | V-SRC |
| MVSep 的「SCNet XL IHF (high instrum fullness by becruily)」 | — | 另一套評測 | MVSep | 與上方不同權重 | — | CONFLICT（同名不同模型，勿混用） |
| **BS-RoFormer**（MUSDB18HQ 版） | 4 stem | — | MUSDB18-HQ | MSST release v1.0.12 | — | V-2ND |
| **MelBand RoFormer guitar**（MVSep） | guitar / other | guitar SDR 由 7.51 → 9.05（更新後） | MVSep guitar 評測 | MVSep 線上，非全部開放權重 | — | V-2ND、日期 CONFLICT（2026-03 或 2026-10） |
| **MVSep Mega 53 Stems** | 53 種樂器（含 electric / acoustic guitar；**無 lead/rhythm 分軌**） | 無統一數字；官方承認單軌品質可能低於專用模型；**stems 相加不等於原混音** | — | MSST release v1.0.21（BS-RoFormer 架構）；建議 ≥16 GB VRAM | 程式 MIT，**權重無明確授權** | V-SRC（release 頁） |
| **SubRoformer** | 4 stem | 論文 11.59 dB（單模型、+500 首額外資料）；MSST 表列 base 約 10.42 dB | MUSDB18-HQ | 官方 repo `ArchiMickey/SubRoformer`（2026-11-06 更新）；MSST key `sub_roformer`，release v1.0.23 標示「08 Nov」 | 未確認 | V-SRC（arXiv 2611.08130）；數字 CONFLICT；「2026-11 新權重」年份 **未能 100% 確認** |
| **HTDemucs 6s**（baseline） | 6 stem | other 2.63（4 stem htdemucs 為 5.57）；guitar/piano 無官方 SDR | MVSep | demucs v4 | MIT | V-2ND；社群普遍評價 piano 不可用、guitar 普通 |
| MVSep Lead/Rhythm Guitar、Electric/Acoustic Guitar | 各 2 stem | 無公開比較 benchmark | — | **僅 MVSep 線上服務，未確認開放權重** | — | V-2ND，不可離線則不納入主流程 |
| FLOMSS（flow matching，2026-10） | 4 stem | 「competitive」，未取得數字 | MUSDB18-HQ | 未確認 | — | UNVERIFIED，觀察名單 |

### 2.2 架構結論

```
mix ──► [A] BS-RoFormer SW (6 stem)  ──► guitar / piano / other / bass / drums / vocals
   └──► [B] SCNet XL IHF 或 SubRoformer (4 stem) ──► drums / bass / vocals / other
   └──► [C] 專用模型（可選）：MelBand guitar、Mega 53 中的 electric/acoustic guitar、synth、keys
```

- drums、bass、vocals：A 與 B 做 **波形平均或逐頻帶選擇（ensemble）**，並計算兩者差異作為該 stem 的信心。
- guitar：以 A 為主；C 存在時與 A 比對。若 A-guitar 與 C-guitar 在某片段能量差異過大 → 該片段標記低信心。
- **other stem 絕不視為 guitar**（硬性規則，寫入單元測試）。
- Mega 53 的輸出不可做 mixture consistency 假設（官方註明 stems 不相加為原混音）。
- 多支吉他（lead / rhythm、左右聲道）：開放權重中**沒有經驗證的 lead/rhythm 分離器**。第一版改用 (1) 立體聲聲像（mid/side、左右聲道相關性）分群，(2) 轉譜後依音域與複音度分群；無法區分時輸出單一「Guitar (combined)」並標記，不得捏造分軌。

---

## 3. 其他子系統調查（使用者需求在 3.1 後中斷，以下為研究者補充）

### 3.1 多樂器 AMT

| 模型 | 重點 | 授權 | 狀態 |
|---|---|---|---|
| **MuScriptor**（Kyutai × Mirelo，2026-07；PyPI 0.4.0，2026-10-20） | decoder-only transformer；small 103M / medium 307M（預設）/ large ≈1.3–1.4B；36 種樂器；可指定樂器條件（如只轉鼓）；可輸出 PDF 樂譜與吉他 TAB；自報 Multi F1 ≈47.8–48.2、Drums F1 ≈49.6–50.2 | 程式 MIT；**權重 CC BY-NC 4.0**，需在 HF 接受授權 | V-SRC（GitHub/HF）；數字 CONFLICT（兩來源略異） |
| **YourMT3+**（2024） | MT3 延伸，MoE、多通道 decoder；後續論文指其在真實錄音表現不佳 | Apache-2.0（UNVERIFIED） | V-SRC |
| MT3 | 2025 AMT Challenge 中仍是強 baseline | Apache-2.0 | V-2ND |
| Basic Pitch（Spotify） | 輕量、單一樂器無關的 note + pitch bend | Apache-2.0 | UNVERIFIED（記憶） |

**決策**：對每個「已分離」的 stem 執行 MuScriptor（以樂器條件限定，例如 guitar stem 只允許 guitar 類樂器），另以 Basic Pitch 做第二意見；兩者 onset/pitch 一致的音符信心高，僅一方偵測到的音符標記低信心或丟棄（依設定）。也對**原始混音**跑一次 MuScriptor 不帶條件，作為樂器歸屬交叉驗證：若混音轉譜把某音歸給 piano，而 guitar stem 轉譜也出現同一音，代表 stem 有洩漏 → 依歸屬投票決定。

### 3.2 吉他 / 貝斯 TAB

| 方案 | 狀態 |
|---|---|
| Noise2Fret（arXiv 2608.30854，ISMIR 2026，diffusion，playability-aware） | 程式碼 / 權重未確認 |
| TART（arXiv 2609.11904，T5 指派弦/格，輸出 beat-aligned MusicXML，含技巧辨識） | 程式碼未確認 |
| Fretting-Transformer / open-fret（MIDI→TAB，T5） | GitHub 有實作，品質 MUST-BENCH |
| MuScriptor 內建 TAB | 可用，但非商業 |
| 規則式 DP（Viterbi，cost = 手指跨度 + 把位移動 + 開放弦獎勵 + 同時發聲弦衝突） | 自行實作，保底方案 |

**決策**：實作 `FretAssigner` 介面，預設 DP 演算法；MuScriptor TAB 與 open-fret 作為可插拔後端。必須支援調弦設定（Standard、Drop D、Eb、7/8 弦、5/6 弦 Bass），**J-Rock / Metal 常見降半音與 Drop 調弦**，以最低音偵測自動建議調弦。

### 3.3 鼓譜

| 方案 | 狀態 |
|---|---|
| ADTOF（Zehren et al.；PyTorch port 分數 88.26 vs 原版 88.68） | V-2ND；權重 CC BY-NC-SA 4.0 |
| ADTOF Plus（MDX23C 分離 + ADTOF + open/closed hi-hat） | V-2ND |
| ADT_STR（arXiv 2601.09520，2026-01，合成資料訓練的 seq2seq，含 velocity） | V-SRC（論文）；MUST-BENCH |
| MuScriptor drums 條件模式 | V-SRC |
| arXiv 2509.24853：先做鼓組子分離可提升 ADT | V-SRC |

**決策**：drums stem → ADTOF 與 MuScriptor(drums) 雙路 → onset 對齊投票 → 映射到 GM drum map（kick, snare, hi-hat open/closed/pedal, toms ×3, crash, ride, china, splash）→ 量化到節拍格。

### 3.4 節拍 / BPM / 拍號 / 調性 / 段落

| 任務 | 方案 | 狀態 |
|---|---|---|
| Beat / downbeat | **Beat This!**（CPJKU，ISMIR 2024，無 DBN 後處理；PyPI `beat-this`） | V-SRC |
| BPM | 由 beat interval 推導，含 tempo map（允許變速） | — |
| 拍號 | downbeat 間 beat 數統計 + 局部變拍偵測（Progressive Rock 需支援 7/8、5/4 等） | 自行實作 |
| 調性 | 第一版：Krumhansl / Temperley profile 套用在轉譜後的 pitch-class histogram（排除鼓）；可選 MIRFLEX key 模型 | UNVERIFIED，MUST-BENCH |
| 和弦 | Chord-CNN-LSTM（ChordMini 所用）為候選 | V-2ND |
| 段落 | `all-in-one`（mir-aidj，beat/downbeat/段落一體）為候選；保底：自相似矩陣 + novelty | UNVERIFIED |

---

## 4. 系統架構

```
bandscribe/
  cli.py                 # bandscribe transcribe song.flac --out out/ --profile quality
  env_check.py           # OS / CUDA / VRAM / ffmpeg / torch 檢測與報告
  io/ audio_load.py      # mp3/wav/flac → 44.1k float32 stereo，保留原取樣率資訊
  analysis/ beats.py tempo.py meter.py key.py sections.py
  separation/
    base.py              # Separator 介面：separate(audio) -> dict[str, Stem(audio, model_id, sr)]
    bs_roformer_sw.py scnet.py subroformer.py mega53.py htdemucs6s.py
    ensemble.py          # 多模型合併 + 每 stem 每時間片段信心
  attribution/
    instrument_id.py     # stem 活動偵測（RMS / 頻譜）、混音轉譜投票、洩漏偵測
  transcription/
    base.py              # Transcriber 介面 -> list[Note(onset, offset, pitch, velocity, conf, source)]
    muscriptor.py basic_pitch.py adtof.py
    consensus.py         # 多模型音符對齊與投票
  tab/ fret_dp.py tunings.py backends/
  notation/
    quantize.py          # 依 tempo map 量化、tuplet 偵測
    score_model.py       # 內部樂譜 IR（與輸出格式無關）
    export_midi.py export_musicxml.py export_gp.py export_pdf.py
  qa/
    checks.py            # 品質規則（第 5 節）
    report.py            # report.json + report.html（低信心片段時間軸）
  models/registry.yaml   # 每個模型：來源 URL、版本、sha256、授權、VRAM 需求
```

### 4.1 執行流程

1. `env_check`：輸出報告；無 CUDA 時允許 CPU 但警告。
2. 載入音訊 → 正規化 → 快取 hash。
3. 節拍 / tempo map / 拍號 / 段落（對原混音與 drums stem 各跑一次並比較）。
4. 分離（第 2.2 節），每 stem 存檔並快取。
5. 樂器歸屬：判斷每 stem 是否真的有該樂器（活動度門檻）；空 stem 不轉譜。
6. 逐 stem 轉譜（多模型）→ consensus。
7. 吉他 / 貝斯 → TAB 指派。鼓 → 鼓譜映射。
8. 量化 → 內部樂譜 IR。
9. QA 檢查 → 低信心片段標記（在 MusicXML 用 text annotation / 顏色，在報告列出時間碼）。
10. 匯出：總譜 + 每樂器分譜（MIDI、MusicXML、GP、PDF）。

### 4.2 輸出格式

| 格式 | 工具 | 備註 |
|---|---|---|
| MIDI | `pretty_midi` / `mido` | 每樂器一軌，含 tempo map |
| MusicXML | `music21` 或自寫 writer | 吉他/貝斯同時輸出五線譜 + TAB staff |
| **Guitar Pro 8（`.gp`）— 主要交付格式** | **alphaTab `Gp7Exporter`**（官方文件：輸出 GP7+ `.gp`，GP7 與 GP8 皆可開啟；自 alphaTab 1.2.0 起） | 詳見第 12.1 節 |
| Guitar Pro 5（`.gp5`，後備） | `PyGuitarPro` 0.10.1 | GP8 亦可開啟 `.gp5`；當 `.gp` 匯出失敗時使用 |
| PDF | MuseScore 4 CLI（`mscore -o out.pdf in.musicxml`）或 LilyPond | 需 env_check 偵測是否安裝；缺少時跳過並提示 |

---

## 5. 品質規則（硬性）

1. **禁止 other → guitar**：other stem 只能經過樂器歸屬判定後才可能產生吉他音符，且必須同時被混音轉譜支持。
2. **禁止捏造**：不得以重複前一小節、插值、或「補滿小節」的方式產生音符。空白小節以休止符表示。
3. 每個音符帶 `conf ∈ [0,1]` 與 `sources`；低於門檻（預設 0.5）者：預設保留但標記；`--strict` 模式移除。
4. 音域檢查：超出樂器 / 調弦可演奏範圍的音 → 移除或歸屬給其他樂器，並記錄。
5. 可演奏性：吉他同時發聲 ≤ 6（或弦數）、手指跨度 ≤ 5 格（可設定）；違反者標記。
6. 貝斯原則上單音，複音片段標記。
7. 鼓：同一時間 > 4 個肢體事件標記（人類只有四肢）。
8. 分離交叉驗證差異大的時間區段 → 所有該 stem 的音符降信心。
9. 報告必須列出：使用的每個模型與版本 / sha256、各 stem 能量、每樂器低信心比例。

---

## 6. 必做 Benchmark（Fable 實作期間）

| 項目 | 資料 | 指標 |
|---|---|---|
| 分離 | MUSDB18-HQ test（4 stem）；MoisesDB（有 guitar/piano 標註） | SDR（museval / `fast_bss_eval`），每模型每 stem；VRAM 峰值；秒/分鐘音訊 |
| 吉他轉譜 | GuitarSet、（可取得時）GOAT | `mir_eval` note F1（onset、onset+offset） |
| TAB | GuitarSet | string/fret 正確率、TDR |
| 鼓 | ENST-Drums 或 STAR Drums 子集、ADTOF test | 每類 onset F1 |
| Beat | GTZAN / Ballroom 子集 | F-measure、downbeat F |
| 端到端 | 使用者自備 3–5 首 J-Rock / Metal 樂曲（含官方 TAB 對照） | 人工審查 + 低信心片段覆蓋率 |

資料集授權各異，下載前列出並請使用者確認。

---

## 7. 環境與相依

- Python 3.11、PyTorch（CUDA 12.x），以 `uv` 管理。
- Windows 原生優先測試；若某模型（如需 Triton / flash-attn）在 Windows 失敗，於 env_check 報告並建議 WSL2。
- 模型權重一律下載至 `~/.cache/bandscribe/models/`，以 registry 的 sha256 驗證；**下載前列出檔名、來源、大小請使用者確認**。
- 所有推論離線可執行（首次下載後）。

---

## 8. 授權風險總表

| 元件 | 授權 | 影響 |
|---|---|---|
| MuScriptor 權重 | CC BY-NC 4.0 | 僅限非商業 |
| ADTOF 權重 | CC BY-NC-SA 4.0 | 僅限非商業 |
| madmom 模型 | CC BY-NC-SA 4.0 | 盡量不用（Beat This! 取代） |
| MVSep Mega 53 權重 | 未標示 | 使用前需確認 |
| BS-RoFormer SW / SubRoformer 權重 | 未確認 | 使用前需確認 |
| HTDemucs、Basic Pitch | MIT / Apache-2.0（UNVERIFIED） | 低風險 |

---

## 9. 未決問題

已由使用者於 2026-11-12 回答（GP8、RTX 3080 10 GB、個人使用、需 GUI），見第 12 節。剩餘：

1. 使用者是否有 Guitar Pro 8 可做最終人工開檔驗證（Fable 無法自動化 GP8 本體）。
2. 端到端測試用的 3–5 首樂曲由使用者提供。

---

## 10. 給 Fable 的實作里程碑

| M | 內容 | 完成定義 |
|---|---|---|
| M0 | env_check、專案骨架、模型 registry、測試框架 | `bandscribe doctor` 輸出完整環境報告；CI 單元測試綠燈 |
| M1 | 音訊載入、Beat This!、tempo map、拍號 | GTZAN 子集 beat F ≥ 公開數值 −3% |
| M2 | 分離：BS-RoFormer SW + SCNet XL IHF + HTDemucs 6s baseline + ensemble | MUSDB18-HQ test 各模型 SDR 表；ensemble 不劣於最佳單模型 |
| M3 | SubRoformer、Mega 53 整合與比較 | 更新 benchmark 表，依實測決定預設 |
| M4 | 樂器歸屬 + 轉譜（MuScriptor、Basic Pitch、ADTOF）+ consensus | GuitarSet note F1 報告；other→guitar 規則測試通過 |
| M5 | TAB DP + 調弦偵測 + 鼓譜映射 | GuitarSet TAB 正確率報告 |
| M6 | 量化、樂譜 IR、MIDI / MusicXML / GP5 / PDF 匯出 | 匯出檔可被 MuseScore 4 與 Guitar Pro 開啟（MuseScore 自動測試） |
| M7 | QA 規則、report.html、端到端 | 使用者樂曲端到端產出完整總譜 + 分譜 + 低信心報告 |

每個里程碑完成後：更新 `BENCHMARKS.md`、紀錄未達標項目，不得以修改測試門檻的方式「通過」。

---

## 12. 補充需求的設計決策（v0.2）

### 12.1 Guitar Pro 8 輸出

**事實（V-SRC，alphaTab 官方文件）**
- GP7 / GP8 的 `.gp` 是 ZIP，內含 `Content/score.gpif`（XML）及 GP6 沿用的 stylesheet / configuration blob。
- alphaTab `Gp7Exporter.export(score, settings)` 回傳 bytes；文件聲明 GP7 與 GP8 皆可開啟；alphaTab 專屬功能會降級為最接近的 GP7 表示。
- alphaTab 可匯入 MusicXML。
- Python 生態**沒有**經驗證可「從零寫出」GP8 `.gp` 的函式庫（PyGuitarPro 僅 GP3–5；`gpif` 僅讀取；Guitar-Pro-Utilities 只做嵌入音訊）。Go 的 `go-guitar-pro` 有 GP8 匯出但官方註明非無損。

**決策**
```
score IR ──► alphaTab 資料模型（經由 MusicXML 或直接產生 alphaTex）
          ──► Node 腳本呼叫 Gp7Exporter ──► song.gp（GP8 開啟）
          └─► 後備：PyGuitarPro ──► song.gp5
```
- 優先路徑：Python 產生 **alphaTex**（alphaTab 的文字格式，可明確指定弦/格、調弦、技巧），由 Node 端 `exporter/export_gp.mjs` 讀入並輸出 `.gp`。MusicXML 匯入作為第二路徑。Fable 需實測哪一條保留的資訊較完整（弦/格、調弦、tempo map、拍號變化、鼓組映射、各軌樂器音色）。
- `.gp` 檔內容：每樂器一軌（吉他軌含 TAB + 五線譜、Bass TAB、Drums 打擊樂譜、Vocals/Piano/Synth 五線譜）、正確調弦、tempo 自動化、段落標記（Section markers）、低信心片段以文字註記（如 `?low-conf`）標示。
- **可選**：將原始音訊嵌入為 backing track（GP8 功能）。參考 Guitar-Pro-Utilities 的 injector 實作邏輯，列為延伸項目，MUST-BENCH。
- 驗證：(1) 自動：用 alphaTab 重新讀回 `.gp`，比對音符數、弦/格、小節數一致；(2) 人工：使用者以 GP8 開檔確認。
- 授權：alphaTab 為 MPL-2.0（UNVERIFIED，Fable 確認）。

### 12.2 RTX 3080 10 GB VRAM 限制

搜尋未找到各分離模型在 10 GB 的官方 VRAM 數據 → 全部 MUST-BENCH。設計規則：
- **模型一次只載入一個**，用完立即釋放（`del model; torch.cuda.empty_cache()`），不並行。
- 分離模型預設 FP16 / autocast，`chunk_size` 與 `batch_size` 可調，並在 OOM 時自動降低 chunk 重試，最後退回 CPU。
- **MVSep Mega 53（官方建議 ≥16 GB）**：預設**停用**，在 GUI 設定中標示「實驗性，可能極慢或失敗」；若實測 FP16 + 小 chunk 能跑則開放。
- MuScriptor：預設 **medium（307M）**；large（≈1.3B）在 FP16 下權重約 2.6 GB，理論上可行但需實測。
- `env_check` 報告每個模型實測的峰值 VRAM（`torch.cuda.max_memory_allocated`），寫入 `BENCHMARKS.md`。
- 品質優先：允許長時間處理（單首 5 分鐘歌曲數十分鐘可接受），GUI 必須顯示分步進度與預估時間。

### 12.3 授權（僅個人使用）

使用者確認僅個人使用 → MuScriptor（CC BY-NC 4.0）、ADTOF（CC BY-NC-SA 4.0）**可列入預設流程**。GUI「關於」頁面與 README 必須列出每個模型的授權，並註明「輸出不可用於商業用途」。

### 12.4 圖形介面

**決策：本地 Web App，包成桌面視窗。**
- 後端：FastAPI（Python，與推論同一環境），任務佇列（單 GPU → 一次一首），WebSocket 推送進度。
- 前端：Vite + React + TypeScript；樂譜預覽與播放直接用 **alphaTab**（與 GP 匯出共用同一函式庫，所見即所得）。
- 桌面化：`pywebview` 開啟本地視窗（Windows 原生）；亦可直接用瀏覽器開 `http://127.0.0.1:<port>`。只綁定 127.0.0.1。
- 頁面：
  1. **匯入**：拖放 MP3/WAV/FLAC，顯示波形與 BPM / 調性 / 拍號初步分析。
  2. **設定**：品質設定檔（quality / balanced）、模型選擇、調弦（自動偵測或手動）、樂器啟用、低信心門檻、strict 模式。
  3. **處理進度**：各階段進度條、目前模型、VRAM 使用、可取消。
  4. **分軌檢視**：各 stem 播放 / solo / mute，顯示每段信心熱圖；使用者可手動把某 stem 標記為「不是吉他」等修正歸屬後重跑後段流程。
  5. **樂譜檢視**：alphaTab 渲染總譜 / 分譜 / TAB，低信心音符以顏色標示，點擊跳到對應音訊位置並 A/B 比對原音。
  6. **匯出**：GP8 `.gp`、`.gp5`、MIDI、MusicXML、PDF、QA 報告；一鍵開啟輸出資料夾。
- 所有處理在 GUI 與 CLI 共用同一 pipeline（GUI 不得有獨立邏輯）。

### 12.5 里程碑調整

| M | 變更 |
|---|---|
| M0 | 加入 Node（alphaTab 匯出器）與前端 toolchain 檢測；VRAM 管理工具 |
| M6 | 以 **GP8 `.gp`** 為主要驗收：alphaTab 讀回一致 + 使用者 GP8 人工開檔通過 |
| M8（新增） | GUI：上述 6 個頁面，pywebview 桌面視窗，端到端以 GUI 完成一首歌 |

---

## 11. 來源

- BS-RoFormer SW：https://mvsep.com/en/algorithms ・ https://mvsep.com/news/64?lang=en ・ https://pypi.org/project/bs-roformer-infer/ ・ https://github.com/openmirlab/bs-roformer-infer ・ https://huggingface.co/Acapellas/bs-roformer-sw-6stem-FP16-FP32-onnx
- SCNet XL IHF：https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md ・ https://github.com/ZFTurbo/Music-Source-Separation-Training/releases ・ https://github.com/openmirlab/scnet-infer ・ https://arxiv.org/pdf/2401.13276
- MSST 框架論文：https://arxiv.org/abs/2607.23395
- 分離對節奏特徵的影響：https://arxiv.org/abs/2609.04224
- SubRoformer：https://arxiv.org/abs/2611.08130 ・ https://github.com/ZFTurbo/Music-Source-Separation-Training/pull/267 ・ https://github.com/ArchiMickey/SubRoformer
- MVSep Mega 53：https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.21
- MelBand RoFormer：https://arxiv.org/pdf/2310.01809 ・ https://mvsep.com/quality_checker/leaderboard/guitar ・ https://mvsep.com/es/news
- HTDemucs 6s：https://huggingface.co/StemSplitio/htdemucs-6s-onnx ・ https://arxiv.org/html/2406.18747v1
- 吉他 lead/rhythm：https://mvsep.com/news/67?lang=en ・ https://moises.ai/blog/moises-news/new-guitar-separation-models
- FLOMSS：https://www.alphaxiv.org/abs/2610.17821
- MuScriptor：https://github.com/muscriptor/muscriptor ・ https://huggingface.co/MuScriptor/muscriptor-large ・ https://pypi.org/project/muscriptor/ ・ https://arxiv.org/html/2607.08168v2 ・ https://www.marktechpost.com/2026/07/10/kyutai-releases-muscriptor-an-open-weight-decoder-only-transformer-for-multi-instrument-music-transcription-to-midi/
- YourMT3+：https://arxiv.org/abs/2407.04822
- TAB：https://arxiv.org/abs/2608.30854v1 ・ https://arxiv.org/pdf/2609.11904 ・ https://github.com/Sidmaz666/open-fret ・ https://github.com/aguilasa/tabgen
- 鼓：https://github.com/xavriley/adtof_plus_drum_transcription ・ https://huggingface.co/thebluescreen/adtof-drums-onnx ・ https://arxiv.org/pdf/2601.09520 ・ https://github.com/pier-maker92/ADT_STR ・ https://arxiv.org/abs/2509.24853
- Beat This!：https://arxiv.org/html/2407.21658v1 ・ https://pypi.org/project/beat-this/
- 和弦 / 調性 / 段落：https://dev.to/ngha_phantrng_ccdda2bf/open-source-chord-beat-detection-application-39f5 ・ https://github.com/DoMusic/U-Net
- PyGuitarPro：https://github.com/Perlence/PyGuitarPro ・ https://pypi.org/project/PyGuitarPro/
- alphaTab GP7/GP8：https://alphatab.net/docs/features/export/export-to-guitar-pro-7 ・ https://alphatab.net/docs/reference/types/exporter/gp7exporter/ ・ https://alphatab.net/docs/features/import-load/guitar-pro ・ https://npmjs.com/package/@coderline/alphatab
- GP8 格式相關專案：https://github.com/mlemczyk/Guitar-Pro-Utilities ・ https://github.com/gwaggott/gp-2-midi ・ https://github.com/CaliLuke/go-guitar-pro ・ https://github.com/teaqu/guitar-pro-youtube-sync
