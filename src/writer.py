import json
from pathlib import Path


def _ts(sec: float) -> str:
    s = int(sec)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _srt_ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_timeline(utterances: list[dict], path: Path) -> None:
    lines = [f"[{_ts(u['start'])}] {u['speaker']}: {u['text']}"
             for u in utterances]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_by_speaker(utterances: list[dict], path: Path) -> None:
    grouped: dict[str, list[dict]] = {}
    for u in utterances:
        grouped.setdefault(u["speaker"], []).append(u)

    blocks = []
    for speaker, items in grouped.items():
        total = sum(u["end"] - u["start"] for u in items)
        header = f"===== {speaker}  （発話 {len(items)}回 / 合計 {_ts(total)}）====="
        body = "\n".join(f"  [{_ts(u['start'])}] {u['text']}" for u in items)
        blocks.append(f"{header}\n{body}")

    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def write_srt(utterances: list[dict], path: Path) -> None:
    chunks = []
    for i, u in enumerate(utterances, 1):
        chunks.append(
            f"{i}\n{_srt_ts(u['start'])} --> {_srt_ts(u['end'])}\n"
            f"{u['speaker']}: {u['text']}\n"
        )
    path.write_text("\n".join(chunks), encoding="utf-8")


def write_json(utterances: list[dict], path: Path) -> None:
    path.write_text(
        json.dumps(utterances, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
