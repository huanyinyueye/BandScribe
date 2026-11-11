# BandScribe QA / 模型比較 / E2E 計畫（Opus session 負責）

分工（使用者指示，2026-11-12）：
- **Fable session**：框架與模型調用（adapter、pipeline、匯出、GUI）。介面見 `docs/INTERFACES.md`。
- **Opus session（本檔）**：模型比較 benchmark、QA 規則、端到端驗證，以及依實測結果維護 `bandscribe/models/defaults.yaml`。

Opus 擁有的路徑：`benchmarks/`、`BENCHMARKS.md`、`tests/e2e/`、`bandscribe/qa/`、`bandscribe/models/defaults.yaml`、`QA_PLAN.md`。
共用 checkout，各自只 `git add` 自己的路徑。

## 1. 資料集（存放於 `~/.cache/bandscribe/datasets/`）

| 資料集 | 用途 | 來源 | 狀態 |
|---|---|---|---|
| GTZAN + beat_this annotations | beat / downbeat | HF `marsyas/gtzan`、CPJKU | 已有（M1） |
| MUSDB18-HQ | 4 stem 分離 SDR | zenodo 3338373 | 下載中 |
| MoisesDB | guitar / piano 分離 SDR | 需註冊，可能無法自動下載 | 待確認；若取不得，以 MUSDB 的 other + 自建 Slakh 混音代替 |
| Slakh2100（子集） | 6 stem（含 guitar、piano）分離與多樂器轉譜 | zenodo | 待下載 |
| GuitarSet | 吉他 note F1、TAB string/fret | zenodo 3371780 | 待下載 |
| ENST-Drums / STAR Drums / ADTOF test | 鼓 per-class onset F1 | 各官方來源 | 待下載 |
| 使用者樂曲 3–5 首 | E2E | 使用者提供 | 待提供 |

## 2. 模型比較

### 2.1 分離（每模型、每 stem）
- 指標：SDR（`museval` 風格 chunked SDR 中位數 + 全曲 SDR，兩者都報）、峰值 VRAM、秒/音訊分鐘、OOM 重試次數。
- 評測集：MUSDB18-HQ test 50 首（drums/bass/vocals/other）；Slakh 子集（guitar/piano）。
- 比較對象：BS-RoFormer SW、SCNet XL IHF、SubRoformer、HTDemucs 6s（baseline）、Mega 53（若 10 GB 可跑）、ensemble 組合。
- 額外檢查：other stem 洩漏到 guitar 的能量比例；mixture consistency 誤差。
- 決策規則：ensemble 必須在每個 stem 上不劣於最佳單模型 0.1 dB 以上，否則該 stem 改用最佳單模型。

### 2.2 轉譜
- 吉他：GuitarSet（以 hex-pickup 混合後的單聲道作輸入）note F1（onset / onset+offset，`mir_eval.transcription`）。
- 多樂器：Slakh 子集 per-instrument F1。
- 比較：MuScriptor small / medium / large、Basic Pitch、consensus。
- 鼓：per-class onset F1，比較 ADTOF、MuScriptor drums、consensus。

### 2.3 TAB
- GuitarSet 真實弦/格 vs DP 指派：string/fret 準確率、TDR、可演奏性違規數。

## 3. QA 規則（`bandscribe/qa/`）
實作規格第 5 節 9 條硬性規則，`run_checks(score_ir, context) -> QAReport`：
音符標記（low_conf / out_of_range / unplayable / limb_overflow / leakage_suspect）、
`--strict` 刪除、報告 JSON + HTML（低信心片段時間軸）。每條規則有單元測試，
包含「other stem 產生吉他音符必須被拒絕」的回歸測試。

## 4. E2E（`tests/e2e/`）
- 合成測試曲：以 Slakh MIDI 渲染的已知答案混音 → 全流程 → 與 ground truth 比對（note F1、TAB 準確率、小節數、拍號、BPM）。
- `.gp` 驗證：alphaTab 讀回後音符數、弦/格、小節數、tempo 一致。最終以使用者 GP8 人工開檔確認。
- 使用者樂曲：產出完整總譜 + 分譜 + QA 報告，人工審查清單。

## 5. 進度

| 項目 | 狀態 |
|---|---|
| 分工訊息送達 Fable | 已送出（2026-11-12） |
| MUSDB18-HQ 下載 | 進行中 |
| 分離 benchmark harness | 待 Fable 的 `get_separator` 介面 |
