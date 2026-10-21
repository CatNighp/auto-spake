# auto_spake — 会議 話者別文字起こし

会議の録画（動画・音声）から、**誰が・いつ・何を話したか**を自動で書き起こすツールです。
処理はすべてローカルで完結し、音声データが外部に送信されることはありません。

- **文字起こし** — mlx-whisper（Apple Metal で高速推論）
- **話者分離** — pyannote.audio 4.x（誰が話しているかを区間で判定）
- **動作環境** — Apple Silicon Mac（M1 以降 / 16 GB RAM 以上推奨）

---

## セットアップ

### 1. 前提ソフト

| ソフト | インストール |
|---|---|
| [Homebrew](https://brew.sh/) | 公式サイト参照 |
| ffmpeg | `brew install ffmpeg` |
| Python 3.12 | [uv](https://docs.astral.sh/uv/) で管理（下記参照） |

### 2. リポジトリの取得と環境構築

```bash
git clone https://github.com/<your-username>/auto_spake.git
cd auto_spake

# uv がなければ先にインストール: https://docs.astral.sh/uv/getting-started/installation/
uv venv --python 3.12
source .venv/bin/activate
uv pip install mlx-whisper "pyannote.audio==4.0.7" torch torchaudio gradio
```

> **注意**: パッケージ追加時は `pip install` ではなく **`uv pip install`** を使ってください。

### 3. HuggingFace の準備（話者分離モデル用）

pyannote のモデルは無料ですが、利用規約への同意が必要です。

1. [HuggingFace](https://huggingface.co/join) でアカウントを作成
2. 以下の各ページで **Agree and access repository** をクリック
   - https://huggingface.co/pyannote/speaker-diarization-community-1
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
3. [Settings → Tokens](https://huggingface.co/settings/tokens) で **Read** トークンを発行
4. CLI でログイン:
   ```bash
   hf auth login
   ```

確認:

```bash
source .venv/bin/activate
python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('pyannote/speaker-diarization-community-1','config.yaml'); print('OK')"
```

`OK` が出れば準備完了です。

---

## 使い方

### GUI（ブラウザ）

```bash
source .venv/bin/activate
python src/app.py
```

ブラウザが自動で開きます（開かない場合は http://127.0.0.1:7860 ）。

1. **会議の録画** に動画または音声ファイルをドラッグ＆ドロップ
2. **参加者の人数** を入力（精度が大きく変わるため、分かっていれば必ず指定）
3. **固有名詞のヒント**（任意）に社名・専門用語を入力すると誤変換が減ります
   例: `BilTec, 与信管理, SFA`
4. **「文字起こしを開始」** を押して待機
5. 完了後、プレビューとダウンロード用ファイルが表示されます

#### 話者に実名をつける

結果の話者は `SPEAKER_00` のような仮名で出力されます。

1. プレビューで発話時間・占有率を見て、どの番号が誰か判断
2. **「表示名」** 欄に実名を入力
3. **「表示名を反映して出力し直す」** を押すと、名前入りで再出力されます

#### 終了

ターミナルで **Ctrl + C** を押すと停止します。

### CLI（コマンドライン）

```bash
source .venv/bin/activate
python src/cli.py path/to/meeting.mp4 --num-speakers 3
```

**主なオプション:**

| オプション | 意味 | 既定値 |
|---|---|---|
| `-n, --num-speakers` | 参加者数（指定推奨） | 自動推定 |
| `-l, --language` | 言語 | `ja` |
| `--prompt` | 固有名詞のヒント（例 `"BilTec, SFA"`） | なし |
| `-m, --model` | Whisper モデル | 高速・高精度モデル |
| `--device` | 話者分離の実行先 `mps` / `cpu` | 自動 |
| `--max-gap` | この秒数以上の無音で発話を区切る | `1.0` |
| `--batch-size` | 話者分離のバッチサイズ（メモリ不足なら下げる） | `32` |
| `-o, --outdir` | 出力先フォルダ | 入力と同じ場所 |

---

## 出力ファイル

`data/output/<ファイル名>/` に以下の形式で保存されます。

| ファイル | 用途 |
|---|---|
| `*.timeline.txt` | **時系列の会話ログ** — 議事録の素材や要約AIへの入力に |
| `*.by_speaker.txt` | **話者ごとの発言集** — 特定の人の発言を一覧したいときに |
| `*.srt` | **字幕ファイル** — VLC / IINA 等で動画と一緒に再生可能 |
| `*.json` | **構造化データ** — 他ツールへの受け渡し用 |
| `*.wav` | 中間ファイル（削除可） |

> **字幕で精度確認:** `.srt` を元動画と同じフォルダ・同じ名前で配置し、
> VLC や IINA で再生すると、話者ラベルの正確さを目と耳で確認できます。

---

## LAN 共有

同じネットワーク上の他のユーザーに共有するには、`src/app.py` の `demo.launch()` を編集します。

```python
demo.launch(
    server_name="0.0.0.0",          # LAN に公開
    auth=("user", "password"),      # パスワード認証（推奨）
    inbrowser=True,
)
```

起動後、`http://<Mac の IP アドレス>:7860` でアクセスできます。

> **注意:** 認証なしで `0.0.0.0` に公開すると、同一ネットワーク上の誰でもアクセスできます。
> Metal とメモリを1処理が占有するため、同時利用は実質1人ずつ（順番待ち）です。

---

## アーキテクチャ

```
録画ファイル
  │
  ├─①  audio.py       ffmpeg で 16kHz モノラル WAV に変換
  │
  ├─②  transcribe.py  mlx-whisper で文字起こし（Metal）
  │
  ├─③  diarize.py     pyannote で話者分離
  │
  ├─④  merge.py       ②と③を時刻で突き合わせ、話者ラベルを付与
  │
  └─    writer.py     4形式で書き出し
```

②（文字起こし）と③（話者分離）は独立して並行実行され、④で結合します。
GUI（`app.py`）と CLI（`cli.py`）は、どちらもこのパイプラインの入口です。

---

## プロジェクト構成

```
auto_spake/
├── src/
│   ├── app.py         GUI（Gradio）
│   ├── cli.py         CLI
│   ├── audio.py       ① 音声抽出
│   ├── transcribe.py  ② 文字起こし
│   ├── diarize.py     ③ 話者分離
│   ├── merge.py       ④ 突き合わせ
│   └── writer.py      出力
├── data/
│   └── output/        結果の保存先
└── README.md
```

---

## トラブルシューティング

| 症状 | 原因・対処 |
|---|---|
| `No module named 'gradio'` 等 | 仮想環境に入っていない → `source .venv/bin/activate` |
| `pip install` が反映されない | `uv pip install <パッケージ名>` を使う |
| `GatedRepoError` / `403` | HuggingFace の規約同意漏れ → 上記3リポジトリすべてに同意済みか確認 |
| 認証エラー / `Pipeline` が `None` | `hf auth login` でログイン済みか確認 |
| 処理が遅い | `--device` を `mps` と `cpu` 両方試す |
| メモリ不足 | `--batch-size` を下げる（32 → 16 → 8） |
| 話者ラベルが途中で入れ替わる | 既知の制限。`--max-gap` の調整や人数指定の見直しで改善する場合がある |
| ブラウザが開かない | http://127.0.0.1:7860 を手動で開く |

---

## ライセンス

MIT
