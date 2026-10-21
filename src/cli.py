import argparse
import time
from pathlib import Path

from audio import extract_audio, get_duration
from transcribe import transcribe, DEFAULT_MODEL
from diarize import diarize
from merge import merge
from writer import write_timeline, write_by_speaker, write_srt, write_json


def main() -> None:
    p = argparse.ArgumentParser(description="会議映像の話者別文字起こし")
    p.add_argument("input", help="動画または音声ファイル")
    p.add_argument("-o", "--outdir", default=None,
                   help="出力先ディレクトリ（既定: 入力と同じ場所）")
    p.add_argument("-n", "--num-speakers", type=int, default=None,
                   help="参加者数。分かっているなら必ず指定する")
    p.add_argument("--min-speakers", type=int, default=None)
    p.add_argument("--max-speakers", type=int, default=None)
    p.add_argument("-m", "--model", default=DEFAULT_MODEL)
    p.add_argument("-l", "--language", default="ja")
    p.add_argument("--prompt", default=None,
                   help="固有名詞のヒント（例: 'BilTec, 与信管理'）")
    p.add_argument("--device", default=None, choices=["mps", "cpu"],
                   help="話者分離の実行デバイス。未指定なら mps を優先")
    p.add_argument("--max-gap", type=float, default=1.0,
                   help="同一話者の発話を分割する無音の長さ（秒）")
    p.add_argument("--batch-size", type=int, default=32,
                   help="話者分離のバッチサイズ。メモリ不足なら下げる")
    args = p.parse_args()

    src = Path(args.input)
    outdir = Path(args.outdir) if args.outdir else src.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = src.stem

    t0 = time.time()

    print("[1/4] 音声を抽出中…")
    wav = extract_audio(src, outdir / f"{stem}.wav")
    dur = get_duration(wav)
    print(f"      {wav.name}  ({dur / 60:.1f} 分)")

    print("[2/4] 文字起こし中（Metal）…")
    t = time.time()
    asr = transcribe(wav, model=args.model, language=args.language,
                     initial_prompt=args.prompt)
    print(f"      完了 {time.time() - t:.1f}秒 / {len(asr['segments'])} セグメント")

    print("[3/4] 話者分離中…")
    t = time.time()
    dia = diarize(wav,
                  num_speakers=args.num_speakers,
                  min_speakers=args.min_speakers,
                  max_speakers=args.max_speakers,
                  device=args.device,
                  batch_size=args.batch_size)
    speakers = sorted({s["speaker"] for s in dia})
    print(f"      完了 {time.time() - t:.1f}秒 / 話者 {len(speakers)}人 {speakers}")

    print("[4/4] 突き合わせ・書き出し中…")
    utterances = merge(asr, dia, max_gap=args.max_gap)

    write_timeline(utterances, outdir / f"{stem}.timeline.txt")
    write_by_speaker(utterances, outdir / f"{stem}.by_speaker.txt")
    write_srt(utterances, outdir / f"{stem}.srt")
    write_json(utterances, outdir / f"{stem}.json")

    print(f"\n完了: {len(utterances)} 発話 / 総処理時間 {time.time() - t0:.1f}秒")
    print(f"出力先: {outdir}")
    for suffix in ("timeline.txt", "by_speaker.txt", "srt", "json"):
        print(f"  - {stem}.{suffix}")


if __name__ == "__main__":
    main()
