import pandas as pd
import numpy as np
import os, numpy as np, pandas as pd, wfdb, torch, torchvision
from tqdm import tqdm
from scipy.signal import resample_poly
from torch import nn
from PIL import Image
import torchvision.transforms as T
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
backbone = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
feature_extractor = nn.Sequential(
    backbone.features,
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(),                            # -> 1024-D
).to(device).eval()

img_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

# ------------------ RESNET50 ------------------
resnet = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
)
resnet_feature = nn.Sequential(
    *(list(resnet.children())[:-1]),   # remove FC → output (2048,1,1)
    nn.Flatten(),                      # → (2048,)
).to(device).eval()

# ---------------- EFFICIENTNET-B0 -------------
effnet = torchvision.models.efficientnet_b0(
    weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
)
effnet_feature = nn.Sequential(
    effnet.features,
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),                      # → (1280,)
).to(device).eval()

# ----------------- CONVNEXT-TINY -------------
convnext = torchvision.models.convnext_tiny(
    weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
)
convnext_feature = nn.Sequential(
    convnext.features,
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(),                      # → (768,)
).to(device).eval()

# ------------- MAIN EMBEDDING FUNCTION --------
def embed_with_all_models(pil_img):

    x = img_tf(pil_img)[None].to(device)

    out = {}

    with torch.no_grad():
        # DenseNet121
        f_dn = feature_extractor(x).squeeze()
        out["densenet"] = torch.nn.functional.normalize(f_dn, p=2, dim=0).cpu().numpy()

        # ResNet50
        f_rs = resnet_feature(x).squeeze()
        out["resnet50"] = torch.nn.functional.normalize(f_rs, p=2, dim=0).cpu().numpy()

        # EfficientNet-B0
        f_eff = effnet_feature(x).squeeze()
        out["efficientnet"] = torch.nn.functional.normalize(f_eff, p=2, dim=0).cpu().numpy()

        # ConvNeXt-Tiny
        f_cn = convnext_feature(x).squeeze()
        out["convnext"] = torch.nn.functional.normalize(f_cn, p=2, dim=0).cpu().numpy()

    return out

# --------------- INTERNAL FUNCTIONS ----------------------------------
def _resolve_abs_base_from_record_id(rid: str):
    base = os.path.join(WAVE_ROOT, rid)
    if os.path.exists(base + ".hea") and os.path.exists(base + ".dat"):
        return base
    if os.path.exists(base + ".hea.gz") and os.path.exists(base + ".dat.gz"):
        return base
    return None

def _load_wfdb_record(abs_base):
    rec = wfdb.rdrecord(abs_base)            #.hea adds .dat
    sig = rec.p_signal.astype(np.float32)    # (n_samples, n_leads)
    fs  = float(rec.fs)
    return sig, fs

def _to_12_leads(sig):
    n, L = sig.shape
    if L == 12: return sig
    out = np.zeros((n, 12), dtype=np.float32)
    out[:, :min(L,12)] = sig[:, :min(L,12)]
    return out

def _fix_length_resample(sig, fs, target_fs=500, target_sec=10):
    if abs(fs - target_fs) > 1e-3:
        sig = resample_poly(sig, up=int(target_fs), down=int(fs), axis=0)
    n_target = int(target_fs * target_sec)
    if sig.shape[0] >= n_target:
        s = (sig.shape[0] - n_target) // 2
        sig = sig[s:s+n_target, :]
    else:
        pad = n_target - sig.shape[0]
        left, right = pad // 2, pad - pad // 2
        sig = np.pad(sig, ((left, right), (0, 0)), mode="constant")
    return sig  # (5000, 12)

def _ecg_to_image(sig_txL):
    # z-score per lead, clip, min-max to 0..255
    x = (sig_txL - sig_txL.mean(axis=0, keepdims=True)) / (sig_txL.std(axis=0, keepdims=True) + 1e-9)
    x = np.clip(x, -5, 5)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) #Handle potential NaNs or Infs
    x = (x - x.min()) / (x.max() - x.min() + 1e-9)
    x = (x * 255.0).astype(np.uint8)         # (5000, 12)
    img = x.T                                # (12, 5000) → HxW
    pil = Image.fromarray(img).resize((224, 224), Image.BILINEAR).convert("RGB")
    return pil

def _embed_record(abs_base):
    sig, fs = load_wfdb_record(abs_base)
    sig = to_12_leads(sig)
    sig = fix_length_resample(sig, fs, 500, 10)
    pil = ecg_to_image(sig)

    feats = embed_with_all_models(pil)

    # return dictionary of numpy arrays
    return {
        "densenet":    feats["densenet"].tolist(),
        "resnet50":    feats["resnet50"].tolist(),
        "efficientnet": feats["efficientnet"].tolist(),
        "convnext":    feats["convnext"].tolist(),
    }

    
# ----------------------OUPUT FUNCTION ---------------------------
def ecg_signal_embedding_extraction(mimiciv_version):
    if not mimiciv_version:
        raise FileNotFoundError("Must run the tabular data (Version 1, 2, or 3) to generate the metadata for the embedding.")

    CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_FILE_DIR))

    WAVE_ROOT = os.path.join(BASE_DIR, "mimiciv", "ecg")
    META_CSV = os.path.join(BASE_DIR, "data", "cohort", f"mimiciv_{mimiciv_version}_cohort.csv.gz")
    OUT_CSV = os.path.join(BASE_DIR, "data", "cohort", "ecg_embeddings.csv.gz")

    df = pd.read_csv(META_CSV).copy()

    #def record_id_from_waveform_path(wp: str) -> str:
    #    return os.path.basename(str(wp).strip("/"))
    df["record_id"] = df["waveform_path"].astype(str).str.strip("/").str.split("/").str[-1]

    df["abs_base"] = df["record_id"].apply(_resolve_abs_base_from_record_id)
    print("Found local files for", df["abs_base"].notna().sum(), "of", len(df), "rows")

    df_match = df[df["abs_base"].notna()].copy().reset_index(drop=True)

    df_match = df[df["abs_base"].notna()].copy().reset_index(drop=True)
    #this abs_base is storing the absolute base path for the local ecg files (both- hea and dat)
    if len(df_match) == 0:
        raise RuntimeError("No matching WFDB files were found. Check WAVE_ROOT and filenames.")

    densenet_emb, resnet_emb, effnet_emb, convnext_emb = [], [], [], []
    mat, miss = 0, 0
    for _, row in tqdm(df_match.iterrows(), total=len(df_match), desc="ECG → embeddings (4 models)"):
        try:
            vecs = _embed_record(row["abs_base"])
            densenet_emb.append(vecs["densenet"])
            resnet_emb.append(vecs["resnet50"])
            effnet_emb.append(vecs["efficientnet"])
            convnext_emb.append(vecs["convnext"])
            mat += 1
        except Exception as e:
            densenet_emb.append(None)
            resnet_emb.append(None)
            effnet_emb.append(None)
            convnext_emb.append(None)
            miss += 1

    df_match["ecg_emb_densenet"] = densenet_emb
    df_match["ecg_emb_resnet50"] = resnet_emb
    df_match["ecg_emb_efficientnet"] = effnet_emb
    df_match["ecg_emb_convnext"] = convnext_emb

    print(f"Embedded {mat}/{len(df_match)}; failed {miss}")

    #keep only the columns needed for the merge (record_id + embedding)
    # removed ecg_map creation and merge as df_match already contains the necessary data.
    #Use df_match directly, which now includes the ecg_emb_dl column
    out = df_match.copy()

    out = out.drop(columns=["abs_base"])
    #out = df.merge(df_match[["abs_base","ecg_emb_dl"]], on="abs_base", how="left")
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved → {OUT_CSV}")