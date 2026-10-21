"""pyannote.audio 4.x による話者分離（いつ・誰が話したかの区間検出）。"""

import os
from pathlib import Path

# MPS が未対応の演算に当たったとき、CPU へ自動退避させる。
# import torch より前に設定する必要がある。
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402
import torchaudio  # noqa: E402
from pyannote.audio import Pipeline  # noqa: E402
from pyannote.audio.pipelines.utils.hook import ProgressHook  # noqa: E402
from pyannote.core import Annotation  # noqa: E402

# pyannote.audio 4.x の標準パイプライン。
# 3.1 とは別リポジトリなので、HuggingFace で個別に規約同意が必要。
MODEL = "pyannote/speaker-diarization-community-1"


def _pick_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_segments(annotation: Annotation) -> list[dict]:
    """pyannote の Annotation を、扱いやすい dict のリストに変換する。"""
    segments = [
        {"start": float(turn.start), "end": float(turn.end), "speaker": speaker}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    segments.sort(key=lambda s: s["start"])
    return segments


def diarize(audio_path: str | Path,
            num_speakers: int | None = None,
            min_speakers: int | None = None,
            max_speakers: int | None = None,
            device: str | None = None,
            batch_size: int = 32,
            exclusive: bool = True,
            show_progress: bool = True) -> list[dict]:
    """話者区間のリストを返す。

    exclusive: True なら重複発話を除いた区間を返す（文字起こしとの
               突き合わせ用）。False なら同時発話を含む生の区間。

    戻り値: [{"start": 0.5, "end": 4.2, "speaker": "SPEAKER_00"}, ...]
    """
    pipeline = Pipeline.from_pretrained(MODEL)
    # ★ pyannote は認証・規約同意に失敗しても例外ではなく None を返す。
    if pipeline is None:
        raise RuntimeError(
            f"{MODEL} を読み込めませんでした。"
            "HuggingFace の規約同意とトークンを確認してください。"
        )

    pipeline.to(_pick_device(device))

    # v4 のデフォルトは両方 1。GPU では極端に非効率なので必ず上げる。
    pipeline.segmentation_batch_size = batch_size
    pipeline.embedding_batch_size = batch_size

    waveform, sample_rate = torchaudio.load(str(audio_path))
    payload = {"waveform": waveform, "sample_rate": sample_rate}

    kwargs: dict = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

    if show_progress:
        with ProgressHook() as hook:
            output = pipeline(payload, hook=hook, **kwargs)
    else:
        output = pipeline(payload, **kwargs)

    # v3 系（および v4 の legacy=True）は Annotation を直接返す
    if isinstance(output, Annotation):
        return _to_segments(output)

    # v4 は DiarizeOutput dataclass
    annotation = (output.exclusive_speaker_diarization if exclusive
                  else output.speaker_diarization)
    return _to_segments(annotation)
