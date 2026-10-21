from typing import Any


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))

def _assign_speaker(start: float, end: float, dia: list[dict]) -> str:
    best_speaker, best_overlap = None, 0.0
    for seg in dia:
        if seg["start"] >= end:
            break
        ov = _overlap(start, end, seg["start"], seg["end"])
        if ov > best_overlap:
            best_speaker, best_overlap = seg["speaker"], ov

    if best_speaker is not None:
        return best_speaker

    if not dia:
        return "UNKNOWN"
    center = (start + end) / 2
    nearest = min(
        dia,
        key=lambda s: 0.0 if s["start"] <= center <= s["end"]
        else min(abs(center - s["start"]), abs(center - s["end"])),
    )
    return nearest["speaker"]


def _iter_words(asr: dict[str, Any]):
    for seg in asr.get("segments", []):
        words = seg.get("words")
        if not words:
            text = seg.get("text", "").strip()
            if text:
                yield float(seg["start"]), float(seg["end"]), text
            continue
        for w in words:
            text = w.get("word", "")
            if not text.strip():
                continue
            # まれに start/end が欠落するので、その場合はセグメント境界で代用
            start = float(w.get("start", seg["start"]))
            end = float(w.get("end", start))
            yield start, end, text


def merge(asr: dict[str, Any], dia: list[dict],
          max_gap: float = 1.0) -> list[dict]:
    utterances: list[dict] = []
    current: dict | None = None

    for start, end, text in _iter_words(asr):
        speaker = _assign_speaker(start, end, dia)

        # 話者が変わった / 間が空きすぎた → 新しい発話を開始
        if (current is None
                or current["speaker"] != speaker
                or start - current["end"] > max_gap):
            if current is not None:
                utterances.append(current)
            current = {"speaker": speaker, "start": start,
                       "end": end, "text": text}
        else:
            current["text"] += text
            current["end"] = end

    if current is not None:
        utterances.append(current)

    for u in utterances:
        u["text"] = u["text"].strip()

    return [u for u in utterances if u["text"]]


def relabel(utterances: list[dict], names: dict[str, str]) -> list[dict]:
    return [{**u, "speaker": names.get(u["speaker"], u["speaker"])}
            for u in utterances]
