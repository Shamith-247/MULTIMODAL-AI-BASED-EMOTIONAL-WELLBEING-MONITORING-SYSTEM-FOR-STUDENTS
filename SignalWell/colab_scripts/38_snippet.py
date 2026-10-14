def label_from_path(path):
    p = path.replace("\\", "/")
    lower = p.lower()
    fname = os.path.basename(p)

    if "ravdess" in lower:
        parts = fname.replace(".wav", "").split("-")
        return RAVDESS_CODE.get(parts[2]) if len(parts) >= 3 else None

    if "crema" in lower:
        parts = fname.replace(".wav", "").split("_")
        return CREMA_CODE.get(parts[2]) if len(parts) >= 3 else None

    if "savee" in lower:
        stem = fname.replace(".wav", "").lower()
        if "_" in stem:
            tail = stem.split("_")[-1]
        else:
            tail = stem
        if tail.startswith("sa"):
            return "sad"
        if tail.startswith("su"):
            return "surprise"
        if tail.startswith("a"):
            return "angry"
        if tail.startswith("d"):
            return "disgust"
        if tail.startswith("f"):
            return "fear"
        if tail.startswith("h"):
            return "happy"
        if tail.startswith("n"):
            return "neutral"
        return None

    if "tess" in lower:
        if "disgust" in lower:
            return "disgust"
        if "fear" in lower:
            return "fear"
        if "happy" in lower:
            return "happy"
        if "neutral" in lower:
            return "neutral"
        if "sad" in lower:
            return "sad"
        if "angry" in lower or "anger" in lower:
            return "angry"
        if "ps" in lower or "surprise" in lower or "pleasant" in lower:
            return "surprise"
        return None

    return None
