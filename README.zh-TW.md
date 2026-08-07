# h3-ui

[English](README.md) | **繁體中文**

在 [vLLM-Omni](https://github.com/vllm-project/vllm-omni) 伺服器上生成影片的輕量瀏覽器前端。

單一檔案、只用標準函式庫、不需編譯。指向一台執行中的伺服器，然後打開瀏覽器就能用。

它存在的理由是：手動對 `/v1/videos/sync` 送請求很煩——multipart 的 body、base64 的附件、一把你不想留在 shell 歷史裡的 API key，以及長到連線一斷就前功盡棄的生成時間。這支程式擋在前面，並把金鑰留在伺服器端。

## 它能做什麼

- **文字、首格圖片、參考條件三種條件模式** — 任務清單直接讀自載入中的 checkpoint，所以 UI 會對齊實際載入的 partition，不會給出伺服器根本會拒絕的選項。
- **送出前就擋掉不合法的附件組合** — 每個任務都標明自己接受什麼，表單會直接拒絕不相符的組合，而不是讓伺服器回 400。
- **真正的佇列** — 想送幾個就送幾個，不必等。單一 worker 依送出順序逐一處理，所以你可以排好一批就走開。排隊中的工作可以取消；正在跑的不行，因為上游呼叫是同步的，而且已經在 GPU 上了。
- **工作不隨頁面消失** — 生成是在伺服器端對著一個 job id 進行，所以關掉分頁或斷線都不會殺掉一個十分鐘的算圖。你回來時完成的影片就在那裡。
- **預設可重現** — 每個結果都會在影片旁寫一份記錄完整參數的 JSON。
- **隨機種子** — 🎲 隨手抽一個；勾選核取方塊則每次生成都抽新的。無論哪種方式，抽到的值都會寫回 seed 欄位並在完成時顯示，所以幸運的結果不會因為記不住數字而消失。
- **結構化 prompt** — H3 要的是三個具名區塊而不是自由文字。表單為每個區塊各留一格，附上格式速查，並依你設定的秒數自動組出 FL2VA 的對齊指令，讓時間戳不會跟表單對不上。你想自己寫的話，純文字一樣可用。
- **刪掉不成功的結果** — 試拍與失敗品可以從歷史紀錄移除，影片與參數檔一起走。

## 需求

- Python 3.10+ — 只用標準函式庫，不必安裝任何東西
- 一台跑著影片模型、連得到的 vLLM-Omni 伺服器

## 安裝設定

```sh
cp .env.example .env
$EDITOR .env          # 設定 H3_API_BASE；伺服器需要金鑰的話再設 H3_API_KEY
python3 server.py
```

然後打開它印出來的網址。

如果你用的是 [MiniMax-H3-DGX-Spark](https://github.com/joeynyc/MiniMax-H3-DGX-Spark) 這套部署 repo，它的 `.env` 會被自動採用，本地的 `.env` 可有可無。

### 設定項目

解析順序：行程環境變數 → `.env` → 預設值。

| 變數 | 預設值 | 意義 |
|---|---|---|
| `H3_API_BASE` | `http://127.0.0.1:8000` | vLLM-Omni base URL（結尾的 `/v1` 會被去掉） |
| `H3_API_KEY` | *(空)* | 有設定時以 `Authorization: Bearer` 送出 |
| `H3_UI_HOST` | `127.0.0.1` | UI 綁定位址 |
| `H3_UI_PORT` | `8080` | UI 連接埠 |
| `H3_UI_ENV_FILE` | *(自動)* | 明確指定 `.env` 的路徑 |

## 語言

介面有英文與繁體中文兩種，依瀏覽器語言自動選擇：**繁體中文語系（`zh-TW` / `zh-Hant` / `zh-HK` / `zh-MO`）看到中文版，其餘語系（包含 `zh-CN`）看到英文版。** 標頭右上角的連結可以手動切換，選擇會記在該瀏覽器的 `localStorage` 裡。

伺服器回傳的錯誤訊息也跟著同一套語言：頁面會在每個 API 請求帶上 `X-Lang`，沒有這個標頭時則回頭看 `Accept-Language`，所以用 `curl` 直接呼叫也會拿到符合你語系的訊息。

## 安全性

**這個行程握有你的 API key，而且瀏覽器要求什麼它就代理什麼。** UI 本身沒有任何驗證——對一個 loopback 工具來說這是刻意的，也正是預設綁在 `127.0.0.1` 的原因。

把 `H3_UI_HOST` 設成 LAN 位址，等於把你 API key 的所有權限交給任何連得到那個埠的人。只在你信任的網路上這麼做；如果只是想從另一台機器用，優先選 SSH tunnel：

```sh
ssh -L 8080:127.0.0.1:8080 user@gpu-box
```

刪除結果會立刻 unlink。沒有垃圾桶，瀏覽器的確認對話框是點下去與檔案消失之間唯一的東西。

## 使用說明

**佇列。** 送出後會加入佇列並回報排隊位置；表單維持可用，所以你可以一次排好幾種變化。佇列面板會顯示等待中的項目、正在跑的項目（已耗時對照預估時間），並且可以取消任何還沒開始的工作。

**重現一個結果。** 每個 `media/*.mp4` 旁邊都有一份 `.mp4.json`：

```json
{
  "task": "t2va",
  "prompt": "Rain on a window at night, soft patter.",
  "width": 768, "height": 448,
  "steps": 20, "duration": 2.0, "fps": 24,
  "flow_shift": 12.0, "audio_flow_shift": 3.0,
  "seed": 43538620,
  "elapsed": 118.6
}
```

把這些值貼回表單——並關掉隨機種子的核取方塊——就會得到同一支影片。

**預估時間**是從一次實測外推出來的，隨 `寬 × 高 × steps × 秒數` 縮放。把它當成數量級，而不是承諾；快取與每步耗時的漂移都會改變實際數字。

## HTTP API

瀏覽器 UI 只是這組 API 的一個客戶端。它能做的事，你都可以用腳本做。

| 方法 | 路徑 | 用途 |
|---|---|---|
| `GET` | `/api/status` | 上游是否可連線、partition、任務清單、佇列深度 |
| `POST` | `/api/generate` | 排入一個工作；回傳 `{id, position}` |
| `GET` | `/api/jobs` | 整個佇列，加上最近完成的工作 |
| `GET` | `/api/job/<id>` | 單一工作的狀態 |
| `POST` | `/api/job/<id>/cancel` | 取消排隊中的工作（已開始則回 409） |
| `GET` | `/api/history` | 已完成的結果與它們的參數 |
| `DELETE` | `/api/history/<file>` | 刪除一個結果與它的參數檔 |
| `GET` | `/media/<file>` | 影片本身 |

所有端點都接受 `X-Lang: en` 或 `X-Lang: zh` 來指定錯誤訊息的語言；沒帶的話會依 `Accept-Language` 判斷。

### Prompt 格式

H3 讀的是三個具名區塊，[官方文件在此](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing)。以 `description`、`soundscape`、`music` 送出，它們會被組成 H3 的欄位名稱：

```text
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium-wide shot frames…

overall_soundscape: Wooden shutters scrape open over a quiet street…

non_diegetic_music: A soft acoustic-guitar pattern at a moderate tempo.
```

`overall_soundscape` 是角色聽得到的聲音；`non_diegetic_music` 是只有觀眾聽得到的配樂——不要配樂就填 `N/A`。鏡頭寫成 `[Shot 1]`，接著 `[Shot 2] At 00:03.500, the camera cuts to…`。運鏡有固定詞彙（`Push In`、`Truck Left`、`Arc Shot`…），可加 `with small amplitude` / `at slow speed` 修飾。對白寫成 `<d>[English] …</d>`，說話者標 `(S1)`。

fl2va 還需要在最前面加一行，說明每張參考圖片落在時間軸的哪裡。這行是從 `duration` 與 description 裡最後一個 `[Shot N]` 產生的，所以時間戳不可能跟你實際要求的秒數不一致：

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the
0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 8.00-second mark…
```

FL2VA 偏好單一鏡頭，好讓模型在兩張圖之間連續內插。

改送純文字的 `prompt` 則會跳過上述全部，原封不動送往上游。

### 請求內容

`POST /api/generate` 收 JSON。`seed` 為負數或不給就代表「幫我抽一個」：

```sh
curl -X POST localhost:8080/api/generate -H 'Content-Type: application/json' -d '{
  "task": "t2va",
  "prompt": "Rain on a window at night, soft patter.",
  "width": 768, "height": 448,
  "steps": 20, "duration": 2.0, "fps": 24,
  "flow_shift": 12, "audio_flow_shift": 3.0,
  "seed": -1,
  "attachments": {}
}'
```

附件是 data URL：`{"image": "data:image/png;base64,..."}`，參考影片條件則用 `{"videos": [...]}`。

## 範圍

伺服器端的契約是 vLLM-Omni 的 `POST /v1/videos/sync`。**任務詞彙**（`t2va`、`fl2va`、`ref2va`）以及從 `model_index.json` 讀 `_minimax_h3` 的 partition 探測是 MiniMax-H3 專屬的——同一台伺服器換一個影片模型，要動的是這兩處，傳輸與工作處理則完全不用改。它不依賴 DGX Spark 或任何特定 GPU。

圖片生成**未**實作。vLLM-Omni 確實有 `/v1/images/generations` 與 `/v1/images/edits`，所以要加的話是多一個任務模式與一塊比較短的結果面板，而不是新的管線；至於能不能生出東西，取決於實際載入的模型。

其他刻意不做的事：沒有使用者帳號、沒有多 GPU 排程、沒有資料庫。工作存在記憶體裡，所以重啟會忘掉佇列——完成的影片在磁碟上，會留著。

## 專案結構

```
server.py       全部：HTTP handler、佇列 worker，以及頁面本身
.env.example    設定範本
media/          生成的影片與它們的參數 JSON（已 gitignore）
```

## 關於長時間生成

vLLM-Omni 用 `_ASYNC_OUTPUT_TIMEOUT` 限制等待某一步背景複製完成的時間，上游把它設成 30 秒。如果單一 denoise 步驟跑得比這久，輸出的 future 會被取消，伺服器的 result-pump 執行緒會死掉——之後 `/health` 還是回 200，但再也不會有任何請求回來。長秒數配高 step 數很容易踩到：在寫這份東西的機器上，50 steps、4 秒的組合每步要跑 44–48 秒。

已回報為 [vllm-project/vllm-omni#5821](https://github.com/vllm-project/vllm-omni/issues/5821)，build 階段的修法在 [joeynyc/MiniMax-H3-DGX-Spark#4](https://github.com/joeynyc/MiniMax-H3-DGX-Spark/pull/4)。把 step 數往上調之前值得先修掉；這個前端沒辦法繞過它。

## 授權

Apache-2.0
