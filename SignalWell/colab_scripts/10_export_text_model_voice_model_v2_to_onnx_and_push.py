# ============================================================
#  EXPORT text_model + voice_model_v2 to ONNX and push to your HF Hub
#  (transformers.js-compatible layout, so the browser can load them)
# ============================================================
!pip -q install "optimum[exporters]" onnx onnxruntime huggingface_hub --upgrade

import os, shutil
from huggingface_hub import HfApi, create_repo, login

# ---- >>> FILL THESE IN <<< ----
HF_TOKEN = "your_hf_write_token_here"        # huggingface.co/settings/tokens -> New token -> Write
HF_USERNAME = "your_hf_username_here"
# --------------------------------

login(token=HF_TOKEN)

from google.colab import drive
drive.mount('/content/drive')
DRIVE_MODELS = "/content/drive/MyDrive/signalwell-models"

def export_and_organize(local_model_dir, task, export_dir):
    os.makedirs(export_dir, exist_ok=True)
    os.system(f"optimum-cli export onnx --model {local_model_dir} --task {task} {export_dir}")
    # transformers.js expects the actual .onnx weights inside an "onnx/" subfolder
    onnx_sub = os.path.join(export_dir, "onnx")
    os.makedirs(onnx_sub, exist_ok=True)
    for f in os.listdir(export_dir):
        if f.endswith(".onnx") and os.path.isfile(os.path.join(export_dir, f)):
            shutil.move(os.path.join(export_dir, f), os.path.join(onnx_sub, f))

# ---- TEXT model ----
print("Exporting text model...")
export_and_organize(f"{DRIVE_MODELS}/text_model", "text-classification", "/content/export_text")

# ---- VOICE model ----
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
print("Use these repo ids in the app: ", TEXT_REPO, "and", VOICE_REPO)
