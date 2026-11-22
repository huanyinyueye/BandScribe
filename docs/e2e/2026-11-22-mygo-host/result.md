# MyGo desktop host smoke run

- Windows: started `go run .` from `desktop/` with the existing project environment and built frontend.
- The process opened a window titled `BandScribe` and started the Python service on an ephemeral loopback port.
- Opened that service in a browser and observed the existing BandScribe import screen.
- Separately, used the real `out/neo-guitar-pc/score.json` in an isolated temporary project with the FastAPI test client: saved an offset and two anchors, read them back, and observed HTTP 400 for decreasing anchor times. This checked persistence, not an interactive alignment flow.
- With a temporary project referencing the real Neo-Aspect audio and existing `.gp`, opened the review workspace in the browser, saw the guitar notation, and started then paused the recording. The displayed position advanced from 0:00 / bar 1 to 0:04 / bar 3. No listening-quality or synchronization accuracy claim is implied.
- On the same project, set the alignment offset to +0.12 seconds, saved it, switched pages, and reopened the review panel. It restored +0.12 and displayed that it was synchronized. This did not test multi-anchor interaction or actual ear-based calibration.
- In an isolated copy of the same score, changed one guitar note's fret through the correction API, exported a separate `.gp`, checked the exporter's alphaTab read-back result, and downloaded the resulting file. This did not open the file in Guitar Pro.
- This smoke run does not verify transcription, editing, audio playback in the desktop window, packaging, or a full guitar review workflow.

## 2026-11-22 前端截圖

- `npm run build` 通過。
- 真實專案 Neo-Aspect（09 Neo-Aspect.wav + out/neo-guitar-pc 轉譜）在 http://127.0.0.1:18767 開啟審譜與改音面板，點選第 2 小節音符可顯示音高／弦／格／位置／時值。
- 截圖：review-editor.png
- 未跑：新增／刪除後儲存、修正版匯出、參考譜上傳、MyGo 視窗內重新截圖。
