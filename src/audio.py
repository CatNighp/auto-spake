import subprocess
from pathlib import Path

def extract_audio(src: str | Path, dst: str | Path | None = None, sample_rate: int = 16000) -> Path:
    src = Path(src)
    if not src.exists():
        raise FileExistsError(f"入力ファイルが見つかりません: {src}")
    
    if dst is None:
        dst = src.with_suffix(".wav")
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd=[
        "ffmpeg" ,"-y",
        "-i", str(src),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(dst),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stderr[-2000:]}")
    return dst

def get_duration(path: str | Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(proc.stdout. strip())
    except ValueError:
        return 