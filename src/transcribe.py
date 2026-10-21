from pathlib import Path
from typing import Any
import mlx_whisper

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"

def transcribe(audio_path: str | Path, model: str = DEFAULT_MODEL, language: str | None = "ja", initial_prompt: str | None = None, verbose: bool = False) -> dict[str, Any]:
    # {
    #     "text": "全文",
    #     "segments": [
    #         {"start": 0.0, "end": 3.2, "text": "...", "words": [{"word": "本日", "start": 0.1, "end": 0.5, "probability": 0.98}, ...]},
    #         ...
    #     ],
    #     "language": "ja",
    # }
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model,
        language=language,
        word_timestamps=True,
        initial_prompt=initial_prompt,
        condition_on_previous_text=False,
        verbose=verbose,
    )
    return result
    