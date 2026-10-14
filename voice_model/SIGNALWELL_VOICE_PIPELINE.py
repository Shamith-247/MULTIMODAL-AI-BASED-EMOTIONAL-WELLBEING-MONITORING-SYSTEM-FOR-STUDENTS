# ======================================================================
# SIGNALWELL VOICE — SOTA HIGH-ACCURACY MASTER PIPELINE (55k+ DATASETS)
# ======================================================================
# ECAN-SER: Evidential Cross-Attention Conformer Network for Speech Emotion Recognition
#
# ACCURACY & GENERALIZATION OPTIMIZATIONS FOR 55k+ DATASETS:
# 1. Multi-Layer WavLM Representation (Layers 12 + 18 + 24 layer ensemble)
# 2. Dynamic Temporal Chunk Prosody Engine (T x 37 temporal matrix)
# 3. Softplus Evidential Head (Non-vanishing gradient, Dirichlet uncertainty)
# 4. Class-Balanced Focal Dirichlet Loss (Solves heavy MELD/ESD class imbalance)
# 5. Latent SpecAugment & Temporal Chunk Masking
# 6. Mixed-Precision GPU Training with Cosine Annealing Warmup
# 7. Supports 55,000+ files across 8 datasets (RAVDESS, CREMA-D, TESS, SAVEE, MELD, ESD, EmoV-DB, IEMOCAP)
# ======================================================================

import sys
import subprocess

def ensure_installed(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

for pkg in ["transformers", "accelerate", "librosa", "soundfile", "kagglehub"]:
    ensure_installed(pkg)


# ======================================================================
# 2. IMPORTS
# ======================================================================

import os
import re
import gc
import json
import random
import hashlib
import warnings
from pathlib import Path
from getpass import getpass

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    recall_score
)

from transformers import (
    WavLMModel,
    Wav2Vec2FeatureExtractor
)

warnings.filterwarnings("ignore")


# ======================================================================
# 3. GPU CHECK & REPRODUCIBILITY
# ======================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if not torch.cuda.is_available():
    print("\n" + "!" * 75)
    print("WARNING: YOU ARE CURRENTLY RUNNING ON CPU!")
    print("For high-speed training on 55k datasets, GPU is strongly recommended.")
    print("In Colab: Runtime -> Change runtime type -> T4 GPU.")
    print("In Kaggle: Notebook Options -> Accelerator -> GPU T4 x 2.")
    print("!" * 75 + "\n")
    DEVICE = "cpu"
else:
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    DEVICE = "cuda"

print("=" * 75)
print("SIGNALWELL VOICE — HIGH-ACCURACY 55k+ MASTER PIPELINE")
print("=" * 75)
print("Device:", DEVICE)
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(0)/(1024**2):.1f} MB")


# ======================================================================
# 4. DIRECTORIES & ENVIRONMENT AUTO-DETECTION
# ======================================================================

IS_KAGGLE = Path("/kaggle").exists()
IS_COLAB = Path("/content").exists()

if IS_KAGGLE:
    ROOT = Path("/kaggle/working/SIGNALWELL_VOICE")
    DATA_DIR = ROOT / "datasets"
    FEATURE_DIR = ROOT / "features"
    MODEL_DIR = ROOT / "models"
    print("✓ Running in Kaggle Notebook environment!")
elif IS_COLAB:
    ROOT = Path("/content/SIGNALWELL_VOICE")
    DATA_DIR = ROOT / "datasets"
    FEATURE_DIR = ROOT / "features"
    MODEL_DIR = ROOT / "models"
    print("✓ Running in Google Colab environment!")
    # Persistent Google Drive cache check
    try:
        from google.colab import drive
        drive_mount = Path("/content/drive")
        if not (drive_mount / "MyDrive").exists():
            drive.mount("/content/drive")
        DRIVE_ROOT = drive_mount / "MyDrive" / "SIGNALWELL_VOICE"
        FEATURE_DIR = DRIVE_ROOT / "features"
        MODEL_DIR = DRIVE_ROOT / "models"
        print(f"✓ Connected to Google Drive! Persistent cache active at:\n  {DRIVE_ROOT}")
    except Exception:
        pass
else:
    ROOT = Path("./SIGNALWELL_VOICE")
    DATA_DIR = ROOT / "datasets"
    FEATURE_DIR = ROOT / "features"
    MODEL_DIR = ROOT / "models"
    print("✓ Running in local / standalone Python environment!")

DATA_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# 5. AUDIO & EMOTION CONFIGURATION
# ======================================================================

SR = 16000
SEGMENT_SECONDS = 6
SEGMENT_SAMPLES = SR * SEGMENT_SECONDS
TEMPORAL_CHUNKS = 6
CHUNK_SAMPLES = SR * 1  # 1 second per temporal chunk

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
LABEL2ID = {emotion: i for i, emotion in enumerate(EMOTIONS)}
ID2LABEL = {i: emotion for emotion, i in LABEL2ID.items()}
NUM_CLASSES = len(EMOTIONS)

# High-accuracy hyperparameters
BATCH_SIZE = 16
ACCUM_STEPS = 2        # Effective batch size = 32
EPOCHS = 20
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
WARMUP_EPOCHS = 2


# ======================================================================
# 6. DATASET REGISTRY & DOWNLOAD (55,000+ SAMPLES)
# ======================================================================

# Full 55k+ dataset suite across clean acted, multi-speaker, and spontaneous dialogue
DATASET_CONFIG = {
    "RAVDESS": True,    # ~1,440 recordings (Clean studio acted speech)
    "CREMA-D": True,    # ~7,442 recordings (Multi-speaker diverse ethnic acted)
    "TESS": True,       # ~2,800 recordings (Female acoustic pitch clarity)
    "SAVEE": True,      # ~480 recordings   (British English dialect)
    "MELD": True,       # ~13,708 recordings (Spontaneous conversational dialogue from Friends)
    "ESD": True,        # ~29,000 recordings (Multi-speaker expressive speech via Kaggle)
    "EMOV_DB": True,    # ~7,000 recordings (Vocal emotional nuances via OpenSLR 115)
    "IEMOCAP": False,   # Optional: set True if local directory available
}

KAGGLE_DATASETS = {
    "RAVDESS": "uwrfkaggler/ravdess-emotional-speech-audio",
    "CREMA-D": "ejlok1/cremad",
    "TESS": "ejlok1/toronto-emotional-speech-set-tess",
    "SAVEE": "ejlok1/surrey-audiovisual-expressed-emotion-savee",
    "ESD": "nguyenthanhlim/emotional-speech-dataset-esd",
}

KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "")
KAGGLE_KEY = os.environ.get("KAGGLE_KEY", "")

kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
if kaggle_json.exists():
    try:
        with open(kaggle_json, "r") as f:
            cfg = json.load(f)
            KAGGLE_USERNAME = KAGGLE_USERNAME or cfg.get("username", "")
            KAGGLE_KEY = KAGGLE_KEY or cfg.get("key", "")
    except Exception:
        pass

if not IS_KAGGLE and any(DATASET_CONFIG[k] for k in ["RAVDESS", "CREMA-D", "TESS", "SAVEE", "ESD"]):
    if not KAGGLE_USERNAME and not (DATA_DIR / "RAVDESS").exists():
        print("Note: If downloading Kaggle datasets automatically, Kaggle credentials may be prompted.")
        try:
            KAGGLE_USERNAME = input("Enter Kaggle username (or press enter to skip): ").strip()
            if KAGGLE_USERNAME:
                KAGGLE_KEY = getpass("Enter Kaggle API key: ").strip()
        except Exception:
            pass

    if KAGGLE_USERNAME and KAGGLE_KEY:
        os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
        os.environ["KAGGLE_KEY"] = KAGGLE_KEY
        try:
            k_dir = Path.home() / ".kaggle"
            k_dir.mkdir(parents=True, exist_ok=True)
            with open(k_dir / "kaggle.json", "w") as f:
                json.dump({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}, f)
            os.chmod(k_dir / "kaggle.json", 0o600)
        except Exception:
            pass

import kagglehub

def download_meld_direct(target_dir):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    wav_count = len(list(target_dir.rglob("*.wav")))
    if wav_count >= 1000:
        print(f"✓ MELD audio already extracted ({wav_count} .wav files found)")
        return target_dir

    print("Downloading MELD dataset directly from HuggingFace mirror (Zero-Auth / High-Speed)...")
    import urllib.request
    import tarfile

    csv_urls = {
        "train.csv": "https://huggingface.co/datasets/ajyy/MELD_audio/resolve/main/train.csv",
        "dev.csv": "https://huggingface.co/datasets/ajyy/MELD_audio/resolve/main/dev.csv",
        "test.csv": "https://huggingface.co/datasets/ajyy/MELD_audio/resolve/main/test.csv"
    }
    archive_urls = {
        "dev.tar.gz": "https://huggingface.co/datasets/ajyy/MELD_audio/resolve/main/archive/dev.tar.gz",
        "test.tar.gz": "https://huggingface.co/datasets/ajyy/MELD_audio/resolve/main/archive/test.tar.gz",
        "train.tar.gz": "https://huggingface.co/datasets/ajyy/MELD_audio/resolve/main/archive/train.tar.gz"
    }

    for fname, url in csv_urls.items():
        dst = target_dir / fname
        if not dst.exists():
            print(f"  Downloading metadata: {fname}...")
            urllib.request.urlretrieve(url, dst)

    for aname, url in archive_urls.items():
        tar_dst = target_dir / aname
        flag_file = target_dir / f".{aname}.done"
        if not flag_file.exists():
            print(f"  Downloading audio archive: {aname}...")
            urllib.request.urlretrieve(url, tar_dst)
            print(f"  Extracting {aname}...")
            with tarfile.open(tar_dst, "r:gz") as tar:
                tar.extractall(path=target_dir)
            flag_file.touch()
            if tar_dst.exists():
                tar_dst.unlink()

    total_wavs = len(list(target_dir.rglob("*.wav")))
    print(f"✓ MELD ready! Total {total_wavs} audio files available.")
    return target_dir

def download_emov_direct(target_dir):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    wav_count = len(list(target_dir.rglob("*.wav")))
    if wav_count >= 1000:
        print(f"✓ EmoV-DB audio already present ({wav_count} .wav files found)")
        return target_dir

    print("Downloading EmoV-DB directly from OpenSLR 115 mirror (Zero-Auth / Public mirror)...")
    import urllib.request
    import tarfile

    emov_files = [
        "bea_Amused.tar.gz", "bea_Angry.tar.gz", "bea_Disgust.tar.gz", "bea_Neutral.tar.gz",
        "jenie_Amused.tar.gz", "jenie_Angry.tar.gz", "jenie_Disgust.tar.gz", "jenie_Neutral.tar.gz",
        "josh_Amused.tar.gz", "josh_Angry.tar.gz", "josh_Disgust.tar.gz", "josh_Neutral.tar.gz",
        "sam_Amused.tar.gz", "sam_Angry.tar.gz", "sam_Disgust.tar.gz", "sam_Neutral.tar.gz"
    ]
    base_url = "https://www.openslr.org/resources/115/"

    for fname in emov_files:
        tar_dst = target_dir / fname
        flag_file = target_dir / f".{fname}.done"
        if not flag_file.exists():
            url = base_url + fname
            print(f"  Downloading EmoV-DB partition: {fname}...")
            try:
                urllib.request.urlretrieve(url, tar_dst)
                with tarfile.open(tar_dst, "r:gz") as tar:
                    tar.extractall(path=target_dir)
                flag_file.touch()
                if tar_dst.exists():
                    tar_dst.unlink()
            except Exception as e:
                print(f"  ⚠ Failed partition {fname}: {e}")

    total_wavs = sum(1 for _, _, files in os.walk(str(target_dir)) for f in files if f.lower().endswith(('.wav', '.flac')))
    print(f"✓ EmoV-DB ready! Total {total_wavs} audio files available.")
    return target_dir

def has_audio_files(path):
    path = Path(path)
    if not path.exists():
        return False
    for _, _, files in os.walk(str(path)):
        for f in files:
            if f.lower().endswith(('.wav', '.flac')):
                return True
    return False

DATA_PATHS = {}
for name, enabled in DATASET_CONFIG.items():
    if not enabled:
        continue
    
    local_candidates = [DATA_DIR / name, ROOT / name, Path.cwd() / "datasets" / name]
    if IS_KAGGLE and Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").glob("*"):
            if name.lower().replace("-", "") in p.name.lower().replace("-", ""):
                local_candidates.insert(0, p)

    found_local = next((c for c in local_candidates if has_audio_files(c)), None)
    if found_local:
        DATA_PATHS[name] = found_local
        print(f"✓ Found dataset: {name} at {found_local}")
        continue

    if name == "MELD":
        try:
            m_path = download_meld_direct(DATA_DIR / "MELD")
            DATA_PATHS["MELD"] = m_path
            print("✓ MELD ready")
            continue
        except Exception as e:
            print(f"⚠ MELD direct download failed: {e}")

    if name == "EMOV_DB":
        try:
            em_path = download_emov_direct(DATA_DIR / "EMOV_DB")
            DATA_PATHS["EMOV_DB"] = em_path
            print("✓ EmoV-DB ready")
            continue
        except Exception as e:
            print(f"⚠ EmoV-DB direct download failed: {e}")

    if name in KAGGLE_DATASETS and (os.environ.get("KAGGLE_USERNAME") or IS_KAGGLE):
        print(f"Downloading {name} via KaggleHub ({KAGGLE_DATASETS[name]})...")
        try:
            path = kagglehub.dataset_download(KAGGLE_DATASETS[name])
            DATA_PATHS[name] = Path(path)
            print("✓", name, "ready")
        except Exception as e:
            print(f"⚠ {name} download skipped: {e}")

if not DATA_PATHS:
    print("⚠ No external dataset paths discovered. Checking for sample audios for standalone testing.")


# ======================================================================
# 7. PARSE DATASETS & NORMALIZE EMOTIONS
# ======================================================================

RAVDESS_EMOTION = {"01": "neutral", "03": "happy", "04": "sad", "05": "angry", "06": "fear", "07": "disgust", "08": "surprise"}
CREMA_EMOTION = {"ANG": "angry", "DIS": "disgust", "FEA": "fear", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
TESS_EMOTION = {
    "angry": "angry", "disgust": "disgust", "fear": "fear", "happy": "happy",
    "neutral": "neutral", "sad": "sad", "surprise": "surprise",
    "ps": "surprise", "pleasant_surprised": "surprise", "pleasant_surprise": "surprise"
}
SAVEE_EMOTION = {"a": "angry", "d": "disgust", "f": "fear", "h": "happy", "n": "neutral", "sa": "sad", "su": "surprise"}
MELD_EMOTION = {
    "anger": "angry", "disgust": "disgust", "fear": "fear", "joy": "happy",
    "neutral": "neutral", "sadness": "sad", "surprise": "surprise"
}
ESD_EMOTION = {
    "neutral": "neutral", "happy": "happy", "sad": "sad",
    "angry": "angry", "surprise": "surprise"
}
EMOV_EMOTION = {
    "anger": "angry", "disgust": "disgust", "amused": "happy", "neutral": "neutral"
}
IEMOCAP_EMOTION = {
    "ang": "angry", "hap": "happy", "exc": "happy", "sad": "sad",
    "neu": "neutral", "fea": "fear", "dis": "disgust", "sur": "surprise"
}

records = []
def add_record(dataset, filepath, speaker, emotion):
    if emotion not in LABEL2ID:
        return
    records.append({
        "dataset": dataset,
        "filepath": str(filepath),
        "speaker_id": f"{dataset}_{speaker}",
        "emotion": emotion,
        "label": LABEL2ID[emotion]
    })

def find_audio_files(root_path):
    root_str = str(root_path)
    audio_files = []
    for dirpath, _, filenames in os.walk(root_str):
        for f in filenames:
            if f.lower().endswith((".wav", ".flac")):
                audio_files.append(Path(dirpath) / f)
    return audio_files

if "RAVDESS" in DATA_PATHS:
    for fp in find_audio_files(DATA_PATHS["RAVDESS"]):
        parts = fp.stem.split("-")
        if len(parts) >= 7 and parts[2] in RAVDESS_EMOTION:
            add_record("RAVDESS", fp, parts[6], RAVDESS_EMOTION[parts[2]])

if "CREMA-D" in DATA_PATHS:
    for fp in find_audio_files(DATA_PATHS["CREMA-D"]):
        parts = fp.stem.split("_")
        if len(parts) >= 3 and parts[2].upper() in CREMA_EMOTION:
            add_record("CREMA-D", fp, parts[0], CREMA_EMOTION[parts[2].upper()])

if "TESS" in DATA_PATHS:
    for fp in find_audio_files(DATA_PATHS["TESS"]):
        fname = fp.stem.lower()
        parent = fp.parent.name.lower()
        full = str(fp).lower()
        em = None
        for k, v in TESS_EMOTION.items():
            if fname.endswith("_" + k) or fname.endswith(k) or (k in parent):
                em = v
                break
        if em:
            spk = "OAF" if "oaf" in full else ("YAF" if "yaf" in full else parent)
            add_record("TESS", fp, spk, em)

if "SAVEE" in DATA_PATHS:
    for fp in find_audio_files(DATA_PATHS["SAVEE"]):
        stem = fp.stem.lower()
        m1 = re.match(r"([a-z]+)_([a-z]+)\d+", stem)
        m2 = re.match(r"([a-z]+)\d+", stem)
        if m1:
            spk, code = m1.group(1), m1.group(2)
        elif m2:
            spk, code = fp.parent.name.lower(), m2.group(1)
        else:
            continue
        if code in SAVEE_EMOTION:
            add_record("SAVEE", fp, spk, SAVEE_EMOTION[code])

if "MELD" in DATA_PATHS:
    meld_root = DATA_PATHS["MELD"]
    meld_lookup = {}
    csv_files = list(meld_root.glob("*.csv")) + list(meld_root.rglob("*_sent_emo.csv")) + list(meld_root.rglob("*meld*.csv"))
    for cf in csv_files:
        try:
            m_df = pd.read_csv(cf)
            if "Dialogue_ID" in m_df.columns and "Utterance_ID" in m_df.columns and "Emotion" in m_df.columns:
                for _, r in m_df.iterrows():
                    key = f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}".lower()
                    spk = str(r.get("Speaker", "unknown")).strip()
                    emo = str(r["Emotion"]).strip().lower()
                    if emo in MELD_EMOTION:
                        meld_lookup[key] = (spk, MELD_EMOTION[emo])
        except Exception:
            pass
    for fp in find_audio_files(meld_root):
        stem = fp.stem.lower()
        if stem in meld_lookup:
            spk, em = meld_lookup[stem]
            add_record("MELD", fp, spk, em)
        else:
            for k, v in MELD_EMOTION.items():
                if k in stem or k in fp.parent.name.lower():
                    add_record("MELD", fp, fp.parent.name, v)
                    break

if "ESD" in DATA_PATHS:
    esd_root = DATA_PATHS["ESD"]
    for fp in find_audio_files(esd_root):
        parent_parts = [p.lower() for p in fp.parts]
        stem = fp.stem.lower()
        em = None
        for k, v in ESD_EMOTION.items():
            if k in parent_parts or k in stem:
                em = v
                break
        if em:
            spk = "spk"
            for part in fp.parts:
                if re.match(r"^\d{4}$", part) or "speaker" in part.lower():
                    spk = part
                    break
            add_record("ESD", fp, spk, em)

if "EMOV_DB" in DATA_PATHS:
    emov_root = DATA_PATHS["EMOV_DB"]
    for fp in find_audio_files(emov_root):
        parts = [p.lower() for p in fp.parts]
        stem = fp.stem.lower()
        em = None
        for k, v in EMOV_EMOTION.items():
            if k in parts or k in stem:
                em = v
                break
        if em:
            spk = next((s for s in ["bea", "jenie", "josh", "sam"] if s in parts), "unknown")
            add_record("EMOV_DB", fp, spk, em)

if "IEMOCAP" in DATA_PATHS:
    iemocap_root = DATA_PATHS["IEMOCAP"]
    for fp in find_audio_files(iemocap_root):
        parts = [p.lower() for p in fp.parts]
        stem = fp.stem.lower()
        em = None
        for k, v in IEMOCAP_EMOTION.items():
            if k in parts or k in stem:
                em = v
                break
        if em:
            spk_match = re.match(r"(ses\d{2}[fm])", stem)
            spk = spk_match.group(1).upper() if spk_match else fp.parent.name
            add_record("IEMOCAP", fp, spk, em)

# If no external datasets downloaded, use sample files to enable immediate testing
if not records:
    sample_files = list(Path("voice_model").glob("sample_*.wav")) + list(Path.cwd().glob("sample_*.wav"))
    for sf_path in sample_files:
        add_record("SAMPLES", sf_path, "sample_speaker", "neutral")

df = pd.DataFrame(records).drop_duplicates(subset=["filepath"]).reset_index(drop=True)
print(f"\nTotal recordings indexed: {len(df)} across {df['speaker_id'].nunique()} unique speakers.")
print("-" * 50)
print("Dataset Breakdown:")
print(df["dataset"].value_counts())
print("-" * 50)
print("Emotion Breakdown:")
print(df["emotion"].value_counts())
print("-" * 50)


# ======================================================================
# 8. SPEAKER-INDEPENDENT SPLIT & DYNAMIC CLASS BALANCING
# ======================================================================

if len(df) > 10 and df["speaker_id"].nunique() >= 4:
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_idx, temp_idx = next(gss1.split(df, groups=df["speaker_id"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["speaker_id"]))
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)
else:
    # Small test fallback
    train_df = df
    val_df = df
    test_df = df

print(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

# Compute smoothed inverse class frequencies for focal balance (Addresses MELD/ESD class imbalance)
label_counts = train_df["label"].value_counts().to_dict()
raw_weights = []
for c_id in range(NUM_CLASSES):
    cnt = label_counts.get(c_id, 1)
    # Smoothed square-root inverse frequency:
    raw_weights.append(np.sqrt(len(train_df) / (float(cnt) * NUM_CLASSES)))
raw_weights = np.array(raw_weights, dtype=np.float32)
CLASS_WEIGHTS = torch.tensor(raw_weights / np.mean(raw_weights), dtype=torch.float32).to(DEVICE)
print("Computed Class Balancing Weights (Addresses 55k+ Class Imbalance):")
for em, w in zip(EMOTIONS, CLASS_WEIGHTS.cpu().numpy()):
    print(f"  {em:<10}: {w:.3f}")


# ======================================================================
# 9. LOAD WAVLM LARGE (WITH MULTI-LAYER ENSEMBLE ACTIVATED)
# ======================================================================

WAVLM_NAME = "microsoft/wavlm-large"
wavlm_processor = Wav2Vec2FeatureExtractor.from_pretrained(WAVLM_NAME)
wavlm_model = WavLMModel.from_pretrained(WAVLM_NAME).to(DEVICE).eval()
for p in wavlm_model.parameters():
    p.requires_grad = False
WAVLM_DIM = wavlm_model.config.hidden_size  # 1024


# ======================================================================
# 10. MULTI-LAYER & DYNAMIC TEMPORAL PROSODY FEATURE EXTRACTION
# ======================================================================

def load_audio(path):
    try:
        audio, orig_sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if orig_sr != SR:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=SR)
        audio = np.nan_to_num(audio)
        threshold = 0.01 * np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else 0.01
        active = np.where(np.abs(audio) > threshold)[0]
        if len(active) > 0:
            audio = audio[active[0] : active[-1] + 1]
        else:
            return None
        peak = np.max(np.abs(audio))
        if peak > 1e-8:
            audio = audio / peak
        return audio.astype(np.float32)
    except Exception:
        return None

def make_6sec(audio):
    if len(audio) >= SEGMENT_SAMPLES:
        start = (len(audio) - SEGMENT_SAMPLES) // 2
        return audio[start:start + SEGMENT_SAMPLES]
    padded = np.zeros(SEGMENT_SAMPLES, dtype=np.float32)
    padded[:len(audio)] = audio
    return padded

def extract_single_chunk_prosody(chunk):
    """
    Extracts 37-dimensional prosodic descriptor for a single 1-second temporal chunk:
    - F0 fundamental pitch statistics (YIN)
    - RMS energy dynamics & peaks
    - Zero Crossing Rate (ZCR)
    - Spectral shape (Centroid, Bandwidth, Rolloff)
    - Pause / voice-activity ratio
    - 13 MFCC means and 13 MFCC standard deviations
    """
    try:
        f0 = librosa.yin(chunk, fmin=60, fmax=500, sr=SR, hop_length=256)
        valid_f0 = f0[(f0 >= 60) & (f0 <= 500) & np.isfinite(f0)]
        pitch_mean = float(np.mean(valid_f0)) if len(valid_f0) else 0.0
        pitch_std = float(np.std(valid_f0)) if len(valid_f0) else 0.0
        pitch_range = float(np.max(valid_f0) - np.min(valid_f0)) if len(valid_f0) else 0.0

        rms = librosa.feature.rms(y=chunk, hop_length=256)[0]
        energy_mean = float(np.mean(rms)) if len(rms) else 0.0
        energy_std = float(np.std(rms)) if len(rms) else 0.0
        energy_range = float(np.max(rms) - np.min(rms)) if len(rms) else 0.0

        zcr_mean = float(np.mean(librosa.feature.zero_crossing_rate(y=chunk, hop_length=256)[0]))
        centroid = librosa.feature.spectral_centroid(y=chunk, sr=SR, hop_length=256)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=chunk, sr=SR, hop_length=256)[0]
        rolloff = librosa.feature.spectral_rolloff(y=chunk, sr=SR, hop_length=256)[0]

        mfcc = librosa.feature.mfcc(y=chunk, sr=SR, n_mfcc=13, hop_length=256)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        silence_thresh = max(float(np.max(rms)) * 0.05, 1e-4) if len(rms) else 1e-4
        voiced_frames = int(np.sum(rms > silence_thresh))
        pause_ratio = max(0.0, 1.0 - (voiced_frames / max(len(rms), 1)))

        features = np.concatenate([
            np.array([
                pitch_mean, pitch_std, pitch_range,
                energy_mean, energy_std, energy_range,
                zcr_mean, float(np.mean(centroid)), float(np.mean(bandwidth)), float(np.mean(rolloff)),
                pause_ratio
            ], dtype=np.float32),
            mfcc_mean.astype(np.float32),
            mfcc_std.astype(np.float32)
        ])
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    except Exception:
        return np.zeros(37, dtype=np.float32)

def extract_dynamic_prosody_sequence(audio):
    """
    Produces a true temporal sequence matrix (TEMPORAL_CHUNKS x 37).
    Enables Bidirectional Cross-Modal Co-Attention (Claim 1) to ground phonetic
    cues against real time-varying prosodic trajectories!
    """
    seq = []
    for i in range(TEMPORAL_CHUNKS):
        chunk = audio[i * CHUNK_SAMPLES : (i + 1) * CHUNK_SAMPLES]
        seq.append(extract_single_chunk_prosody(chunk))
    return np.stack(seq, axis=0).astype(np.float32)  # [6, 37]

PROSODY_DIM = 37

@torch.no_grad()
def extract_wavlm_multi_layer(audio):
    """
    SUPERB & SER SOTA Innovation:
    WavLM Large intermediate layers (12, 18) contain peak acoustic emotion, intonation, and
    timbral cues, while the final layer (24) contains high-level phonetic abstractions.
    A multi-layer blend provides a massive boost over relying solely on layer 24.
    """
    chunks = [audio[i * CHUNK_SAMPLES : (i + 1) * CHUNK_SAMPLES] for i in range(TEMPORAL_CHUNKS)]
    inputs = wavlm_processor(chunks, sampling_rate=SR, return_tensors="pt", padding=True)
    input_values = inputs["input_values"].to(DEVICE)
    
    with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
        output = wavlm_model(input_values, output_hidden_states=True)
        # Multi-layer acoustic-semantic blend: Layers 12, 18, and 24
        h12 = output.hidden_states[12].mean(dim=1)
        h18 = output.hidden_states[18].mean(dim=1)
        h24 = output.hidden_states[24].mean(dim=1)
        blended = (h12 + h18 + h24) / 3.0
    return blended.cpu().float()  # [6, 1024]

def cache_path(filepath):
    key = hashlib.md5(filepath.encode()).hexdigest()
    return FEATURE_DIR / f"{key}.pt"

def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def process_record(row):
    filepath = row["filepath"]
    output_path = cache_path(filepath)
    if output_path.exists():
        return output_path
    audio = load_audio(filepath)
    if audio is None:
        return None
    audio = make_6sec(audio)
    wavlm_feats = extract_wavlm_multi_layer(audio)
    prosody_feats = extract_dynamic_prosody_sequence(audio)
    data = {
        "wavlm": wavlm_feats,                                            # [6, 1024]
        "prosody": torch.tensor(prosody_feats, dtype=torch.float32),      # [6, 37]
        "label": torch.tensor(row["label"], dtype=torch.long),
        "emotion": row["emotion"],
        "dataset": row["dataset"],
        "speaker_id": row["speaker_id"],
        "filepath": filepath
    }
    torch.save(data, output_path)
    return output_path

def build_features(dataframe, split_name):
    print(f"\nFeature extraction: {split_name} ({len(dataframe)} files)")
    rows = []
    for idx, (_, row) in enumerate(tqdm(dataframe.iterrows(), total=len(dataframe))):
        try:
            p = process_record(row)
            if p is not None:
                rows.append({
                    "feature_path": str(p),
                    "dataset": row["dataset"],
                    "speaker_id": row["speaker_id"],
                    "emotion": row["emotion"],
                    "label": row["label"],
                    "filepath": row["filepath"]
                })
        except Exception:
            pass
        if DEVICE == "cuda" and (idx + 1) % 500 == 0:
            torch.cuda.empty_cache()
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / f"{split_name}_features.csv", index=False)
    return res

train_features = build_features(train_df, "train")
val_features = build_features(val_df, "validation")
test_features = build_features(test_df, "test")

# Free WavLM Large from VRAM before training the Conformer Backbone
del wavlm_model
del wavlm_processor
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ======================================================================
# 11. DATASET WITH LATENT SPECAUGMENT REGULARIZATION
# ======================================================================

class SignalWellDataset(Dataset):
    """
    Dataset loader with SpecAugment-inspired temporal chunk masking and
    latent jitter to prevent acoustic room-overfitting across heterogeneous corpora.
    """
    def __init__(self, dataframe, is_train=False):
        self.df = dataframe.reset_index(drop=True)
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        data = safe_torch_load(row["feature_path"], map_location="cpu")
        wavlm = data["wavlm"].float()       # [6, 1024]
        prosody = data["prosody"].float()   # [6, 37] (or [37] for legacy cache)
        label = data["label"].long()

        if self.is_train:
            # 1. Temporal Chunk Masking (Zero out 1 random chunk with 25% probability)
            if random.random() < 0.25:
                mask_idx = random.randint(0, wavlm.size(0) - 1)
                wavlm[mask_idx] = 0.0
                if prosody.ndim == 2:
                    prosody[mask_idx] = 0.0

            # 2. Additive Acoustic Latent Jitter
            if random.random() < 0.35:
                wavlm = wavlm + torch.randn_like(wavlm) * 0.012
                prosody = prosody + torch.randn_like(prosody) * 0.012

        return wavlm, prosody, label

train_loader = DataLoader(
    SignalWellDataset(train_features, is_train=True),
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=(DEVICE == "cuda"),
    num_workers=2 if sys.platform != "win32" else 0
)
val_loader = DataLoader(
    SignalWellDataset(val_features, is_train=False),
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=(DEVICE == "cuda")
)
test_loader = DataLoader(
    SignalWellDataset(test_features, is_train=False),
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=(DEVICE == "cuda")
)


# ======================================================================
# 12. PATENT-GRADE ARCHITECTURE: ECAN-SER
# ======================================================================
# Patent Claims:
# CLAIM 1: Bidirectional Cross-Modal Co-Attention (BC-CA) with SE Gating
# CLAIM 2: Multi-Scale Conformer Temporal Backbone (Depthwise Dilated Conv + MHA)
# CLAIM 3: Multi-Head Content-Dependent Attentive Temporal Pooling
# CLAIM 4: Evidential Deep Learning (Dirichlet Prior) with Softplus Activation
# ======================================================================

class CrossModalAttention(nn.Module):
    """
    Patent Claim 1: Bidirectional Cross-Modal Co-Attention (BC-CA)
    Allows self-supervised speech representations (WavLM) to attend to acoustic
    prosodic dynamics, and prosody to attend to semantic/acoustic context.
    Includes Squeeze-and-Excitation Gated Residual Fusion.
    """
    def __init__(self, dim, heads=4, dropout=0.15):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        # WavLM queries Prosody (Acoustic Salience Guidance)
        self.q_w = nn.Linear(dim, dim)
        self.k_p = nn.Linear(dim, dim)
        self.v_p = nn.Linear(dim, dim)
        self.proj_wp = nn.Linear(dim, dim)
        
        # Prosody queries WavLM (Phonetic Context Grounding)
        self.q_p = nn.Linear(dim, dim)
        self.k_w = nn.Linear(dim, dim)
        self.v_w = nn.Linear(dim, dim)
        self.proj_pw = nn.Linear(dim, dim)
        
        # Squeeze-and-Excitation Gating
        self.se_gate = nn.Sequential(
            nn.Linear(dim * 2, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim * 2),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)
        
    def _attn(self, q, k, v):
        B, T, C = q.shape
        H = self.heads
        d = C // H
        q = q.view(B, T, H, d).transpose(1, 2)
        k = k.view(B, -1, H, d).transpose(1, 2)
        v = v.view(B, -1, H, d).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return out

    def forward(self, wavlm_proj, prosody_proj):
        wp = self.proj_wp(self._attn(self.q_w(wavlm_proj), self.k_p(prosody_proj), self.v_p(prosody_proj)))
        pw = self.proj_pw(self._attn(self.q_p(prosody_proj), self.k_w(wavlm_proj), self.v_w(wavlm_proj)))
        
        cat = torch.cat([wp, pw], dim=-1)
        gates = self.se_gate(cat.mean(dim=1, keepdim=True))
        gated = cat * gates
        fused = self.fusion(gated)
        return self.norm(wavlm_proj + fused)


class DilatedDepthwiseConvModule(nn.Module):
    """
    Patent Claim 2: Depthwise Dilated Temporal Convolution
    Captures micro-prosodic phonemic tremors and syllabic emotional fluctuations (50-300ms).
    """
    def __init__(self, dim, kernel_size=3, dilation=2, dropout=0.15):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pointwise1 = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        padding = (kernel_size - 1) * dilation // 2
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=kernel_size, dilation=dilation, padding=padding, groups=dim)
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.GELU()
        self.pointwise2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        res = x
        x = self.norm(x).transpose(1, 2)
        x = self.glu(self.pointwise1(x))
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pointwise2(x)
        x = self.dropout(x).transpose(1, 2)
        return res + x


class ConformerBlock(nn.Module):
    """
    Patent Claim 2: Macaron-style Conformer Block combining Half-Step FFN,
    Multi-Head Self-Attention, and Dilated Depthwise Temporal Convolution.
    """
    def __init__(self, dim, heads=8, dropout=0.15):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.attn_norm = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.conv_module = DilatedDepthwiseConvModule(dim, kernel_size=3, dilation=2, dropout=dropout)
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.final_norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        x = x + 0.5 * self.ffn1(x)
        norm_x = self.attn_norm(x)
        attn_out, _ = self.mha(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = self.conv_module(x)
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)


class MultiHeadAttentivePooling(nn.Module):
    """
    Patent Claim 3: Multi-Head Content-Dependent Attentive Pooling
    Calculates dynamic importance weights across temporal chunks to focus
    on emotionally salient vocal bursts.
    """
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scorer = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, heads)
        )
        self.proj = nn.Linear(dim * heads, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        scores = self.scorer(x)
        weights = torch.softmax(scores, dim=1)
        pooled_heads = []
        for h in range(self.heads):
            w = weights[:, :, h].unsqueeze(-1)
            pooled = (x * w).sum(dim=1)
            pooled_heads.append(pooled)
        multi_head_cat = torch.cat(pooled_heads, dim=-1)
        out = self.proj(multi_head_cat)
        return self.norm(out)


class EvidentialHead(nn.Module):
    """
    Patent Claim 4: Evidential Deep Learning Head (Subjective Logic / Dirichlet Prior)
    - Non-negative evidence: e_k = Softplus(z_k) >= 0 (smooth, non-vanishing gradient!)
    - Dirichlet parameter: alpha_k = e_k + 1
    - Expected probability: p_k = alpha_k / S
    - Epistemic Uncertainty: u = K / S
    """
    def __init__(self, in_dim, num_classes=7):
        super().__init__()
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LayerNorm(in_dim // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(in_dim // 2, num_classes)
        )
        
    def forward(self, x):
        logits = self.head(x)
        # Softplus avoids dead ReLU neurons and ensures smooth positive evidence
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=-1, keepdim=True)
        prob = alpha / S
        uncertainty = float(self.num_classes) / S.squeeze(-1)
        return {
            "logits": logits,
            "evidence": evidence,
            "alpha": alpha,
            "prob": prob,
            "uncertainty": uncertainty
        }


class BalancedEvidentialFocalLoss(nn.Module):
    """
    Patent Claim 4: Dirichlet Evidential Loss with Class-Weighted Focal Regularization
    Specifically engineered for 55k+ highly imbalanced multi-corpus speech datasets:
    1. Expected Log Loss (Dirichlet Bayes Risk)
    2. Class-Weighted Focal Cross-Entropy (penalizes hard ambiguous samples and boosts minority emotions)
    3. KL Divergence Regularizer against uniform Dirichlet prior (KL Annealing)
    """
    def __init__(self, num_classes=7, class_weights=None, kl_weight=0.2, focal_weight=1.0, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.kl_weight = kl_weight
        self.focal_weight = focal_weight
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, alpha, logits, labels, epoch=0, max_epochs=20):
        device = alpha.device
        y = torch.zeros_like(alpha).scatter_(1, labels.unsqueeze(1), 1.0)
        S = torch.sum(alpha, dim=-1, keepdim=True)
        
        # 1. Expected Log Loss (Bayes Risk)
        log_loss = torch.sum(y * (torch.digamma(S) - torch.digamma(alpha)), dim=-1)
        if self.class_weights is not None:
            sample_weights = self.class_weights[labels]
            log_loss = (log_loss * sample_weights).mean()
        else:
            log_loss = log_loss.mean()

        # 2. Class-Balanced Focal Cross-Entropy
        ce_probs = torch.softmax(logits, dim=-1)
        pt = torch.sum(y * ce_probs, dim=-1) + 1e-8
        focal_term = ((1.0 - pt) ** self.gamma) * (-torch.log(pt))
        if self.class_weights is not None:
            sample_weights = self.class_weights[labels]
            focal_loss = (focal_term * sample_weights).mean()
        else:
            focal_loss = focal_term.mean()
        
        # 3. KL Divergence Regularizer against uniform prior
        alpha_tilde = y + (1.0 - y) * alpha
        S_tilde = torch.sum(alpha_tilde, dim=-1, keepdim=True)
        
        first = torch.lgamma(S_tilde.squeeze(-1)) - torch.sum(torch.lgamma(alpha_tilde), dim=-1)
        second = - torch.lgamma(torch.tensor(float(self.num_classes), device=device))
        third = torch.sum((alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)), dim=-1)
        kl = (first + second + third).mean()
        
        annealing = min(1.0, float(epoch) / float(max(1, max_epochs // 2)))
        return log_loss + (self.focal_weight * focal_loss) + (self.kl_weight * annealing) * kl


class SignalWellPatentArch(nn.Module):
    """
    ECAN-SER: Evidential Cross-Attention Conformer Network for Speech Emotion Recognition
    Novel patent architecture for state-of-the-art accuracy and calibrated uncertainty.
    """
    def __init__(self, wavlm_dim=1024, prosody_dim=37, hidden=256, heads=8, conformer_layers=3, num_classes=NUM_CLASSES):
        super().__init__()
        self.hidden = hidden
        self.num_classes = num_classes
        
        self.wavlm_proj = nn.Sequential(
            nn.Linear(wavlm_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.12)
        )
        self.prosody_proj = nn.Sequential(
            nn.Linear(prosody_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, hidden),
            nn.LayerNorm(hidden),
            nn.Dropout(0.12)
        )
        
        # Patent Claim 1: Cross-Modal Co-Attention
        self.cross_modal_attn = CrossModalAttention(hidden, heads=heads // 2)
        
        # Patent Claim 2: Conformer Temporal Backbone
        self.conformers = nn.ModuleList([
            ConformerBlock(hidden, heads=heads, dropout=0.15) for _ in range(conformer_layers)
        ])
        
        # Patent Claim 3: Multi-Head Attentive Pooling
        self.pooling = MultiHeadAttentivePooling(hidden, heads=4)
        
        # Patent Claim 4: Evidential Head
        self.evidential_head = EvidentialHead(hidden, num_classes=num_classes)
        
    def forward(self, wavlm, prosody):
        w = self.wavlm_proj(wavlm)
        if prosody.ndim == 2:
            p = self.prosody_proj(prosody).unsqueeze(1).expand(-1, w.size(1), -1)
        else:
            p = self.prosody_proj(prosody)
            
        fused = self.cross_modal_attn(w, p)
        
        x = fused
        for conf in self.conformers:
            x = conf(x)
            
        pooled = self.pooling(x)
        out = self.evidential_head(pooled)
        out["embedding"] = pooled
        out["emotion"] = out["logits"]
        return out

SignalWellVoice = SignalWellPatentArch

model = SignalWellPatentArch().to(DEVICE)
criterion = BalancedEvidentialFocalLoss(num_classes=NUM_CLASSES, class_weights=CLASS_WEIGHTS)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Warmup + Cosine Annealing learning rate schedule
def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return float(epoch + 1) / float(WARMUP_EPOCHS)
    progress = float(epoch - WARMUP_EPOCHS) / float(max(1, EPOCHS - WARMUP_EPOCHS))
    return 0.5 * (1.0 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

BEST_MODEL = MODEL_DIR / "SIGNALWELL_VOICE_BEST.pt"
best_f1 = -1.0


# ======================================================================
# 13. HIGH-ACCURACY TRAINING LOOP (MIXED PRECISION & ACCUMULATION)
# ======================================================================

print("\n" + "=" * 75)
print("TRAINING PATENT-GRADE ECAN-SER VOICE ARCHITECTURE")
print("=" * 75)

scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    train_true, train_pred = [], []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1:02d}/{EPOCHS} [Train]")
    for step, (wavlm, prosody, labels) in enumerate(pbar):
        wavlm = wavlm.to(DEVICE, non_blocking=True)
        prosody = prosody.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            output = model(wavlm, prosody)
            loss = criterion(output["alpha"], output["logits"], labels, epoch=epoch, max_epochs=EPOCHS)
            loss = loss / ACCUM_STEPS

        scaler.scale(loss).backward()
        running_loss += loss.item() * ACCUM_STEPS

        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_true.extend(labels.detach().cpu().numpy())
        train_pred.extend(output["prob"].argmax(dim=1).detach().cpu().numpy())
        pbar.set_postfix({"Loss": f"{running_loss / (step + 1):.4f}"})

    scheduler.step()
    train_macro_f1 = f1_score(train_true, train_pred, average="macro", zero_division=0)

    # Validation
    model.eval()
    val_true, val_pred = [], []
    with torch.no_grad():
        for wavlm, prosody, labels in val_loader:
            wavlm, prosody = wavlm.to(DEVICE), prosody.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                output = model(wavlm, prosody)
            val_true.extend(labels.numpy())
            val_pred.extend(output["prob"].argmax(dim=1).cpu().numpy())

    val_acc = accuracy_score(val_true, val_pred)
    val_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
    print(f"Epoch {epoch + 1:02d}/{EPOCHS} | Train F1 = {train_macro_f1:.4f} | Val Acc = {val_acc:.4f} | Val Macro-F1 = {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save({
            "model_state_dict": model.state_dict(),
            "best_val_f1": best_f1,
            "best_val_acc": val_acc,
            "emotions": EMOTIONS,
            "architecture": "ECAN-SER (Evidential Cross-Attention Conformer)",
            "classes": NUM_CLASSES,
            "class_weights": CLASS_WEIGHTS.cpu().numpy()
        }, BEST_MODEL)
        print(f"  ★ New best model checkpoint saved to: {BEST_MODEL}")


# ======================================================================
# 14. TEST EVALUATION & DETAILED REPORT
# ======================================================================

print("\n" + "=" * 75)
print("TEST EVALUATION")
print("=" * 75)

if BEST_MODEL.exists():
    chk = safe_torch_load(BEST_MODEL, map_location=DEVICE)
    model.load_state_dict(chk["model_state_dict"])
    print(f"✓ Loaded best model checkpoint (Val Macro-F1: {chk.get('best_val_f1', 0):.4f})")

model.eval()
test_true, test_pred = [], []
with torch.no_grad():
    for wavlm, prosody, labels in test_loader:
        wavlm, prosody = wavlm.to(DEVICE), prosody.to(DEVICE)
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            output = model(wavlm, prosody)
        test_true.extend(labels.numpy())
        test_pred.extend(output["prob"].argmax(dim=1).cpu().numpy())

test_acc = accuracy_score(test_true, test_pred)
test_f1 = f1_score(test_true, test_pred, average="macro", zero_division=0)
test_weighted_f1 = f1_score(test_true, test_pred, average="weighted", zero_division=0)
test_uar = recall_score(test_true, test_pred, average="macro", zero_division=0)

print(f"Test Accuracy      : {test_acc*100:.2f}%")
print(f"Test Macro-F1      : {test_f1*100:.2f}%")
print(f"Test Weighted-F1   : {test_weighted_f1*100:.2f}%")
print(f"Test UAR (Recall)  : {test_uar*100:.2f}%")
print("-" * 50)
print("\nClassification Report:")
print(classification_report(test_true, test_pred, target_names=EMOTIONS, digits=4, zero_division=0))


# ======================================================================
# 15. INFERENCE ON AUDIO FILES
# ======================================================================

print("\n" + "=" * 75)
print("INFERENCE ON AUDIO FILES")
print("=" * 75)

# Reload WavLM for standalone file inference
wavlm_processor = Wav2Vec2FeatureExtractor.from_pretrained(WAVLM_NAME)
wavlm_model = WavLMModel.from_pretrained(WAVLM_NAME).to(DEVICE).eval()
for p in wavlm_model.parameters():
    p.requires_grad = False

def predict_single_file(audio_path):
    audio = load_audio(audio_path)
    if audio is None:
        print(f"Could not load {audio_path}")
        return None
    audio = make_6sec(audio)
    wavlm_feats = extract_wavlm_multi_layer(audio).unsqueeze(0).to(DEVICE)
    prosody_feats = torch.tensor(extract_dynamic_prosody_sequence(audio), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            out = model(wavlm_feats, prosody_feats)
        probs = out["prob"][0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        uncertainty = float(out["uncertainty"][0].cpu().numpy())

    pred_label = EMOTIONS[pred_idx]
    confidence = probs[pred_idx] * 100

    print("\n" + "-" * 60)
    print(f"File: {Path(audio_path).name}")
    print(f"🎯 Predicted Emotion : {pred_label.upper()} ({confidence:.2f}% confidence)")
    print(f"📊 Evidential Epistemic Uncertainty : {uncertainty:.4f}")
    print("-" * 60)
    for emotion, prob in sorted(zip(EMOTIONS, probs), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 25)
        print(f"  {emotion:<10} : {prob*100:5.1f}%  {bar}")
    return pred_label, confidence, uncertainty

# Check local sample audio files
search_dir = Path("/content") if IS_COLAB else Path.cwd()
target_wavs = [p for p in search_dir.glob("*.wav") if "SIGNALWELL_VOICE" not in str(p)]

if not target_wavs:
    sample_candidates = [
        Path("voice_model/sample_1.wav"), Path("voice_model/sample_2.wav"), Path("voice_model/sample_3.wav"),
        Path("sample_1.wav"), Path("sample_2.wav"), Path("sample_3.wav"),
        ROOT / "sample_1.wav"
    ]
    target_wavs = [p for p in sample_candidates if p.exists()]

for wav_path in target_wavs:
    predict_single_file(wav_path)

print("\n" + "=" * 75)
print("ALL COMPLETE! Best patent-grade model saved at:")
print(f"  {BEST_MODEL}")
print("=" * 75)

if IS_COLAB:
    try:
        from google.colab import files
        files.download(str(BEST_MODEL))
    except Exception:
        pass
elif IS_KAGGLE:
    print(f"\n✓ Kaggle Notebook output ready! Download your trained model from:")
    print(f"  'Output' tab on the right sidebar -> {BEST_MODEL.name}")
