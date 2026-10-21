"""会議映像の話者別文字起こし — ブラウザGUI。

起動:
    .venv/bin/python src/app.py
"""

from pathlib import Path

import gradio as gr

from audio import extract_audio, get_duration
from transcribe import transcribe, DEFAULT_MODEL
from diarize import diarize
from merge import merge, relabel
from writer import write_timeline, write_by_speaker, write_srt, write_json

# 出力先: プロジェクト直下の data/output/<ファイル名>/
OUTDIR = Path(__file__).resolve().parent.parent / "data" / "output"

PREVIEW_LINES = 80  # 画面プレビューに表示する行数


def _hms(sec: float) -> str:
    s = int(sec)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _write_all(utterances: list[dict], outdir: Path, stem: str) -> list[str]:
    """4形式すべてを書き出し、パスのリストを返す。"""
    paths = {
        "timeline.txt": write_timeline,
        "by_speaker.txt": write_by_speaker,
        "srt": write_srt,
        "json": write_json,
    }
    written = []
    for suffix, fn in paths.items():
        p = outdir / f"{stem}.{suffix}"
        fn(utterances, p)
        written.append(str(p))
    return written


def _speaker_table(utterances: list[dict]) -> list[list]:
    """話者ごとの統計を、名前編集用の表データとして組み立てる。"""
    agg: dict[str, list] = {}
    for u in utterances:
        a = agg.setdefault(u["speaker"], [0, 0.0])
        a[0] += 1
        a[1] += u["end"] - u["start"]

    total = sum(v[1] for v in agg.values()) or 1.0
    rows = []
    for speaker, (count, sec) in sorted(agg.items()):
        rows.append([speaker, count, _hms(sec),
                     f"{sec / total * 100:.1f}%", ""])
    return rows


def _preview(utterances: list[dict]) -> str:
    lines = [f"[{_hms(u['start'])}] {u['speaker']}: {u['text']}"
             for u in utterances[:PREVIEW_LINES]]
    if len(utterances) > PREVIEW_LINES:
        lines.append(f"\n…（全 {len(utterances)} 発話中 {PREVIEW_LINES} 件を表示）")
    return "\n".join(lines)


def process(media, num_speakers, language, prompt, model,
            device, max_gap, batch_size,
            progress=gr.Progress()):
    """アップロードされたファイルを処理して、結果一式を返す。"""
    if not media:
        raise gr.Error("動画または音声ファイルをアップロードしてください。")

    src = Path(media)
    stem = src.stem
    outdir = OUTDIR / stem
    outdir.mkdir(parents=True, exist_ok=True)

    progress(0.02, desc="音声を抽出中…")
    wav = extract_audio(src, outdir / f"{stem}.wav")
    minutes = get_duration(wav) / 60

    progress(0.10, desc=f"文字起こし中…（{minutes:.0f}分の音声）")
    asr = transcribe(wav, model=model,
                     language=language or None,
                     initial_prompt=prompt or None)

    progress(0.55, desc="話者分離中…")
    dia = diarize(wav,
                  num_speakers=int(num_speakers) if num_speakers else None,
                  device=device or None,
                  batch_size=int(batch_size),
                  show_progress=False)

    progress(0.92, desc="突き合わせ・書き出し中…")
    utterances = merge(asr, dia, max_gap=float(max_gap))
    files = _write_all(utterances, outdir, stem)

    speakers = sorted({u["speaker"] for u in utterances})
    status = (f"完了　{minutes:.1f}分の音声 / "
              f"話者 {len(speakers)}人 / {len(utterances)} 発話\n"
              f"出力先: {outdir}")

    return (utterances,                      # state に保持
            {"outdir": str(outdir), "stem": stem},
            status,
            _speaker_table(utterances),
            _preview(utterances),
            files,
            gr.update(visible=True))         # 話者名の編集パネルを表示


def apply_names(table, utterances, meta):
    """表に入力された表示名を反映し、ファイルを書き直す。"""
    if not utterances:
        raise gr.Error("先に文字起こしを実行してください。")

    # Gradio の Dataframe は pandas / list のどちらかで渡ってくる
    rows = table.values.tolist() if hasattr(table, "values") else list(table)
    names = {str(r[0]): str(r[4]).strip() for r in rows if str(r[4]).strip()}
    if not names:
        raise gr.Error("「表示名」の列に名前を入力してください。")

    renamed = relabel(utterances, names)
    outdir = Path(meta["outdir"])
    files = _write_all(renamed, outdir, meta["stem"])

    return (renamed,
            f"{len(names)}人の名前を反映しました。",
            _speaker_table(renamed),
            _preview(renamed),
            files)


with gr.Blocks(title="会議 話者別文字起こし") as demo:

    # 処理結果をコールバック間で持ち回るための保持領域
    st_utterances = gr.State([])
    st_meta = gr.State({})

    gr.Markdown(
        "# 会議 話者別文字起こし\n"
        "会議の録画をアップロードすると、**誰が・いつ・何を話したか**を"
        "書き起こします。すべてこのPC内で処理され、外部には送信されません。"
    )

    with gr.Row():
        # ---------------- 左: 入力と設定 ----------------
        with gr.Column(scale=1):
            media = gr.File(
                label="① 会議の録画（動画・音声）",
                file_types=["video", "audio"],
                type="filepath",
            )
            num_speakers = gr.Number(
                label="② 参加者の人数",
                value=3, precision=0, minimum=1, maximum=20,
                info="分かっている場合は必ず入力してください。精度が大きく変わります。",
            )
            prompt = gr.Textbox(
                label="③ 固有名詞のヒント（任意）",
                placeholder="例: BilTec, 与信管理, SFA",
                info="社名・製品名・専門用語を入れると誤変換が減ります。",
            )
            run_btn = gr.Button("文字起こしを開始", variant="primary", size="lg")

            with gr.Accordion("詳細設定", open=False):
                model = gr.Dropdown(
                    label="モデル",
                    choices=[
                        ("高速・高精度（推奨）", DEFAULT_MODEL),
                        ("最高精度・低速", "mlx-community/whisper-large-v3-mlx"),
                    ],
                    value=DEFAULT_MODEL,
                )
                language = gr.Dropdown(
                    label="言語",
                    choices=[("日本語", "ja"), ("英語", "en"), ("自動判定", "")],
                    value="ja",
                )
                device = gr.Radio(
                    label="話者分離の実行デバイス",
                    choices=[("自動", ""), ("GPU (Metal)", "mps"), ("CPU", "cpu")],
                    value="",
                )
                max_gap = gr.Slider(
                    label="発話を区切る無音の長さ（秒）",
                    minimum=0.2, maximum=5.0, value=1.0, step=0.1,
                )
                batch_size = gr.Slider(
                    label="バッチサイズ",
                    minimum=1, maximum=64, value=32, step=1,
                    info="メモリ不足で落ちる場合は下げてください。",
                )

        # ---------------- 右: 結果 ----------------
        with gr.Column(scale=2):
            status = gr.Textbox(label="状態", interactive=False, lines=2)

            speaker_table = gr.Dataframe(
                label="話者一覧（「表示名」に実名を入力できます）",
                headers=["ラベル", "発話回数", "発話時間", "占有率", "表示名"],
                datatype=["str", "number", "str", "str", "str"],
                column_count=(5, "fixed"),
                static_columns=[0, 1, 2, 3],
                interactive=True,
            )

            name_row = gr.Row(visible=False)
            with name_row:
                apply_btn = gr.Button("表示名を反映して出力し直す")

            preview = gr.Textbox(
                label="会話ログ（プレビュー）",
                lines=22, max_lines=22, interactive=False,
                buttons=["copy"],
            )
            files = gr.File(label="ダウンロード", file_count="multiple")

    gr.Markdown(
        "### 使い方のヒント\n"
        "- 話者は `SPEAKER_00` のような仮のラベルで出ます。"
        "プレビューを読んで誰か分かったら、上の表の「表示名」に入力し"
        "「反映して出力し直す」を押してください。\n"
        "- `.srt` を元の動画と同じフォルダに置くと、VLC や IINA で"
        "字幕として再生しながら精度を確認できます。\n"
        "- 90分の会議で15分程度かかります。初回はモデルの"
        "ダウンロード（約1.7GB）が入るため追加で時間がかかります。"
    )

    run_btn.click(
        fn=process,
        inputs=[media, num_speakers, language, prompt, model,
                device, max_gap, batch_size],
        outputs=[st_utterances, st_meta, status,
                 speaker_table, preview, files, name_row],
    )

    apply_btn.click(
        fn=apply_names,
        inputs=[speaker_table, st_utterances, st_meta],
        outputs=[st_utterances, status, speaker_table, preview, files],
    )


if __name__ == "__main__":
    demo.queue()  # 長時間処理を扱うため必須
    demo.launch(
        theme=gr.themes.Soft(),
        server_name="127.0.0.1",  # 同一LANに公開するなら "0.0.0.0"
        inbrowser=True,
    )
