from huggingface_hub import HfApi

api = HfApi()
print("ログインユーザー:", api.whoami()["name"])

for repo in ["pyannote/speaker-diarization-3.1", "pyannote/segmentation-3.0"]:
    try:
        api.model_info(repo)
        print(f"OK   {repo}")
    except Exception as e:
        print(f"NG   {repo} -> {type(e).__name__}: {e}")