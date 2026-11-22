# 匯出改由 Go 執行（Phase 1）E2E

日期：2026-11-22。`bandscribe/notation/export_{gp,gp5,midi,musicxml}.py` 改成呼叫
`bin/bandscribe-export.exe`（`cmd/bandscribe-export`），函式簽名與回傳的 report 不變，
`pipeline.py` 與 GUI 的修正匯出不用改。

## 實際跑了什麼

1. `pipeline._export` 對四首真實歌曲（`out/{neo-guitar-pc,ringing-guitar-pc,singalive,songiam}/score.json`）
   匯出 `.gp`、`.gp5`、`.mid`、`.musicxml`。執行時把 Node 從 PATH 拿掉，並封鎖 `guitarpro`、`mido` 匯入。
   腳本：`.cache/e2e-go-exports/run.py`。
   - 四首都成功；`.gp` 全部 `verified: true`、`mismatches: []`，`alphatab: 1.9.0`。
   - 每首四種格式合計 0.95–1.22 秒（四次子程序啟動）。
2. 同四首用舊版 Python 匯出器（commit `1788a40`，`git archive` 取出）重匯出比對
   （`.cache/e2e-go-exports/check.py`）：
   - MIDI、MusicXML、`.gp5` 逐 byte 相同；`.gp` zip 內每個檔案相同、score.gpif 元素樹相同。
   - alphaTab 讀回 `.gp`：小節數與音符數正確（例：Neo-Aspect 174 小節、5452 音符）。
   - pyguitarpro 讀回 `.gp5`、mido 讀回 `.mid` 都成功。
3. `scripts/parity_exports.py --stress 300`（參考版改由 git 取出）：304 個案例中 `.gp`、`.gp5`、
   MusicXML 全部一致；MIDI 187 一致，其餘 117 是 Python 遇到非 Latin-1 文字會失敗、Go 照常寫檔（刻意的差異）。
4. `.gp5` 對抗案例 32/32 一致（弦號 0/7/-1/8、fret ±、力度 95、分母 1/3/32/64/128、15 軌、
   第一軌 8 弦被跳過、cp1252 特殊字元、超長字串），包含 Python 會報錯的輸入兩邊都拒絕。
5. Repo verify：`ruff check bandscribe` 通過；`pytest -q` 219 過、1 失敗、1 略過。
   失敗的是 `test_pipeline_end_to_end_output_structure`：它斷言分軌 `device == "cuda"`，
   另一個 session 的 audio.cpp 分軌現在回報 `"cuda (audio.cpp)"`，與匯出無關（該次流程 status=done）。

## 沒驗證的

- 本機沒有 Guitar Pro 8 / MuseScore，沒有在 GP8 實際開檔；以 alphaTab 讀回代替。
- `pyproject.toml` 還沒把 `pyguitarpro`、`mido` 移到開發依賴（檔案有另一個 session 未提交的改動）；
  `env_check` 也還會檢查 Node / alphaTab（同理，等對方提交）。
