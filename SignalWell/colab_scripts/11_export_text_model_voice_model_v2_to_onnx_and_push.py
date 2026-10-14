# ============================================================
#  EXPORT text_model + voice_model_v2 to ONNX and push to your HF Hub
#  (fixed: uses optimum-onnx, the package HF split ONNX export into)
# ============================================================
!pip uninstall -y onnxruntime -q
!pip install -q "optimum-onnx[onnxruntime]" huggingface_hub --upgrade

import os, shutil
from huggingface_hub import HfApi, create_repo, login

# ---- >>> FILL THESE IN <<< ----
HF_TOKEN = "your_hf_write_token_here"
HF_USERNAME = "Shamith247"
# --------------------------------

login(token=HF_TOKEN)

from google.colab import drive
drive.mount('/content/drive')
DRIVE_MODELS = "/content/drive/MyDrive/signalwell-models"

def export_and_organize(local_model_dir, task, export_dir):
    if not os.path.exists(local_model_dir):
        raise FileNotFoundError(f"Can't find {local_model_dir} — check it actually exists in Drive")
    os.makedirs(export_dir, exist_ok=True)
    exit_code = os.system(f"optimum-cli export onnx --model {local_model_dir} --task {task} {export_dir}")
    if exit_code != 0:
        raise RuntimeError(f"Export command failed (exit code {exit_code}) for {local_model_dir} — check the printed error above")

    onnx_files = [f for f in os.listdir(export_dir) if f.endswith(".onnx")]
    if not onnx_files:
        raise RuntimeError(f"No .onnx file was produced in {export_dir} — export did not actually succeed")
    print(f"  -> produced: {onnx_files}")

    onnx_sub = os.path.join(export_dir, "onnx")
    os.makedirs(onnx_sub, exist_ok=True)
    for f in onnx_files:
        shutil.move(os.path.join(export_dir, f), os.path.join(onnx_sub, f))

print("Exporting text model...")
export_and_organize(f"{DRIVE_MODELS}/text_model", "text-classification", "/content/export_text")

print("Exporting voice model...")
export_and_organize(f"{DRIVE_MODELS}/voice_model_v2", "audio-classification", "/content/export_voice")

# ---- Push both to the Hub ----
api = HfApi()
TEXT_REPO = f"{HF_USERNAME}/signalwell-text-emotion"
VOICE_REPO = f"{HF_USERNAME}/signalwell-voice-emotion"

create_repo(TEXT_REPO, exist_ok=True)
create_repo(VOICE_REPO, exist_ok=True)

api.upload_folder(folder_path="/content/export_text", repo_id=TEXT_REPO)
api.upload_folder(folder_path="/content/export_voice", repo_id=VOICE_REPO)

print("\n================ DONE ================")
print("Text model live at:  https://huggingface.co/" + TEXT_REPO)
print("Voice model live at: https://huggingface.co/" + VOICE_REPO)
