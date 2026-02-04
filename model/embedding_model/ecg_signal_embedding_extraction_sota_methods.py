import pandas as pd
import numpy as np
import os, numpy as np, pandas as pd, wfdb, torch, torchvision
from tqdm import tqdm
from scipy.signal import resample_poly
from torch import nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import torch
import torchvision.transforms as T
import timm
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

from transformers import AutoImageProcessor, AutoModel
import torch

dino_processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-small"
)
dino_model = AutoModel.from_pretrained(
    "facebook/dinov2-small"
).to(device).eval()

# ================================
# SOTA MODELS
# ================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# DINOv2 ViT-Small
dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
dino_model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()


#  Swin Transformer Small (768-D)

swin = timm.create_model(
    "swin_small_patch4_window7_224",
    pretrained=True,
    num_classes=0
).to(device).eval()

# ConvNeXt Base (1024-D)
convnext_base = timm.create_model(
    "convnext_base",
    pretrained=True,
    num_classes=0
).to(device).eval()

# EfficientNet-V2 Small (1280-D)
effnetv2s = timm.create_model(
    "tf_efficientnetv2_s_in21k",
    pretrained=True,
    num_classes=0
).to(device).eval()


# --------------- INTERNAL FUNCTIONS ----------------------------------
def _resolve_abs_base_from_record_id(rid: str, WAVE_ROOT):
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

def _embed_with_all_models(pil_img):

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
        f_effb0 = effnet_feature(x).squeeze()
        out["efficientnet_b0"] = torch.nn.functional.normalize(f_effb0, p=2, dim=0).cpu().numpy()

        # ConvNeXt-Tiny
        f_cnt = convnext_feature(x).squeeze()
        out["convnext_tiny"] = torch.nn.functional.normalize(f_cnt, p=2, dim=0).cpu().numpy()

        # DINOv2
        inputs = dino_processor(images=pil_img, return_tensors="pt").to(device)
        feats = dino_model(**inputs).last_hidden_state[:, 0, :]
        out["dinov2"] = torch.nn.functional.normalize(feats.squeeze(), p=2, dim=0).cpu().numpy()

        # Swin Small
        f_swin = swin(x).squeeze()
        out["swin_small"] = torch.nn.functional.normalize(f_swin, p=2, dim=0).cpu().numpy()

        # ConvNeXt Base
        f_cnb = convnext_base(x).squeeze()
        out["convnext_base"] = torch.nn.functional.normalize(f_cnb, p=2, dim=0).cpu().numpy()

        # EffNetV2-S
        f_e2 = effnetv2s(x).squeeze()
        out["effnetv2_s"] = torch.nn.functional.normalize(f_e2, p=2, dim=0).cpu().numpy()

    return out


def _embed_record(abs_base):
    sig, fs = _load_wfdb_record(abs_base)
    sig = _to_12_leads(sig)
    sig = _fix_length_resample(sig, fs, 500, 10)
    pil = _ecg_to_image(sig)

    feats = _embed_with_all_models(pil)

    return feats

# ----------------------OUPUT FUNCTION ---------------------------
def ecg_signal_embedding_extraction_sota_methods():
    
    CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))

    BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_FILE_DIR))
    WAVE_ROOT = f"{BASE_DIR}/mimiciv/ecg" #path for the signal files
    META_CSV  = f"{BASE_DIR}/data/cohort/mimiciv_ecg_cohort.csv.gz"
    OUT_CSV = f"{BASE_DIR}/data/cohort/ecg_embeddings_sota_methods.csv.gz"

    df = pd.read_csv(META_CSV).copy()
    #def record_id_from_waveform_path(wp: str) -> str:
    #    return os.path.basename(str(wp).strip("/"))
    df["record_id"] = df["path"].astype(str).str.strip("/").str.split("/").str[-1]

    df["abs_base"] = df["study_id"].astype(str).apply(lambda x: _resolve_abs_base_from_record_id(x, WAVE_ROOT))
    print("Found local files for", df["abs_base"].notna().sum(), "of", len(df), "rows")

    df_match = df[df["abs_base"].notna()].copy().reset_index(drop=True)

    #take only with those that match with the downloaded signals
    df_match = df[df["abs_base"].notna()].copy().reset_index(drop=True)
    #this abs_base is storing the absolute base path for the local ecg files (both- hea and dat)
    if len(df_match) == 0:
        raise RuntimeError("No matching WFDB files were found. Check WAVE_ROOT and filenames.")

    densenet_emb = []
    resnet_emb = []
    effb0_emb = []
    convnext_tiny_emb = []
    dinov2_emb = []
    swin_emb = []
    convnext_base_emb = []
    effv2s_emb = []

    mat, miss = 0, 0

    for _, row in tqdm(df_match.iterrows(),total=len(df_match),desc="ECG → embeddings (8 models)"):
        try:
            vecs = _embed_record(row["abs_base"])

            densenet_emb.append(vecs["densenet"])
            resnet_emb.append(vecs["resnet50"])
            effb0_emb.append(vecs["efficientnet_b0"])
            convnext_tiny_emb.append(vecs["convnext_tiny"])

            dinov2_emb.append(vecs["dinov2"])
            swin_emb.append(vecs["swin_small"])
            convnext_base_emb.append(vecs["convnext_base"])
            effv2s_emb.append(vecs["effnetv2_s"])

            mat += 1
        except Exception as e:
            densenet_emb.append(None)
            resnet_emb.append(None)
            effb0_emb.append(None)
            convnext_tiny_emb.append(None)

            dinov2_emb.append(None)
            swin_emb.append(None)
            convnext_base_emb.append(None)
            effv2s_emb.append(None)

            miss += 1

    df_match["emb_densenet"]       = densenet_emb
    df_match["emb_resnet50"]       = resnet_emb
    df_match["emb_efficientnet_b0"]= effb0_emb
    df_match["emb_convnext_tiny"]  = convnext_tiny_emb

    df_match["emb_dinov2"]         = dinov2_emb
    df_match["emb_swin_small"]     = swin_emb
    df_match["emb_convnext_base"]  = convnext_base_emb
    df_match["emb_effnetv2_s"]     = effv2s_emb

    print(f"Embedded {mat}/{len(df_match)}; failed {miss}")

    #keep only the columns needed for the merge (record_id + embedding)
    # removed ecg_map creation and merge as df_match already contains the necessary data.

    #Use df_match directly, which now includes the ecg_emb_dl column
    out = df_match.copy()
    out = out.drop(columns=["abs_base"])

    #out = df.merge(df_match[["abs_base","ecg_emb_dl"]], on="abs_base", how="left")
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved → {OUT_CSV}")