import os, torch, pandas as pd, numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn
import os

transform1 = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform2 = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


class CXRDataset1(Dataset):
    def __init__(self, df, transform1=None):
        self.df = df.reset_index(drop=True); self.t = transform1
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        img = Image.open(self.df.loc[i, "filepath"]).convert("RGB")
        return (self.t(img) if self.t else img), i

class CXRDataset2(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.t = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img = Image.open(self.df.loc[i, "filepath"]).convert("RGB")
        return (self.t(img) if self.t else img), i
        
device = "cuda" if torch.cuda.is_available() else "cpu"
backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
feature_extractor = nn.Sequential(
    backbone.features,
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(),
).to(device).eval()



# --------------- INTERNAL FUNCTIONS -----------------------------
def _find_path_for_dicom(IMAGES_DIR, dicom_id):
    #exact name
    candidates = [
        os.path.join(IMAGES_DIR, f"{dicom_id}.jpg"),
        os.path.join(IMAGES_DIR, f"{dicom_id}.jpeg"),
        os.path.join(IMAGES_DIR, f"{dicom_id}.png"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    for f in os.listdir(IMAGES_DIR):
        if f.lower().startswith(str(dicom_id).lower()):
            return os.path.join(IMAGES_DIR, f)
    return None

def _find_correct_path(IMAGES_DIR, dicom_id):
    for ext in ["jpg", "jpeg", "png"]:
        p = os.path.join(IMAGES_DIR, f"{dicom_id}.{ext}")
        if os.path.exists(p):
            return p
    return None

def _build_feature_extractor(model_name):
    """Returns a feature extractor (no classification head) and output dimension."""

    if model_name == "densenet121":
        backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        extractor = nn.Sequential(
            backbone.features,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        dim = 1024

    elif model_name == "resnet101":
        backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        extractor = nn.Sequential(
            *(list(backbone.children())[:-1]),
            nn.Flatten()
        )
        dim = 2048

    elif model_name == "efficientnet_b0":
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        extractor = nn.Sequential(
            backbone.features,
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        dim = 1280

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return extractor.to(device).eval(), dim
# ----------------------OUPUT FUNCTION ---------------------------
def mimic_cxr_embedding_FLP(mimiciv_version):
    CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))

    BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_FILE_DIR))
    IMAGES_DIR = os.path.join(BASE_DIR, "mimiciv", "cxr")
    OUT_CSV = os.path.join(BASE_DIR, 'data', 'cohort', 'mimic_cxr_embedding_FLP.csv.gz')

    all_files = os.listdir(IMAGES_DIR)
    img_files = [f for f in all_files if f.lower().endswith(('.jpg','.jpeg','.png'))]

    dicom_ids = [f.rsplit('.', 1)[0] for f in img_files]

    meta = pd.DataFrame({"dicom_id": dicom_ids})


    meta["filepath"] = meta["dicom_id"].apply(lambda x: _find_correct_path(x, IMAGES_DIR))
    #meta["filepath"] = meta["dicom_id"].apply(lambda x: _find_correct_path(IMAGES_DIR,x))
    meta = meta.dropna(subset=["filepath"]).reset_index(drop=True)

    loader = DataLoader(CXRDataset1(meta, transform1), batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Total valid images found: {len(meta)}")

    emb_list = [None] * len(meta)
    with torch.no_grad():
        for x, idxs in tqdm(loader, desc="Embedding CXR (per dicom_id)"):
            x = x.to(device)
            f = feature_extractor(x)                         # (B,1024)
            f = torch.nn.functional.normalize(f, p=2, dim=1)
            for j, irow in enumerate(idxs.tolist()):
                emb_list[irow] = f[j].cpu().numpy().tolist()

    out = meta.copy()
    out["img_embed"] = emb_list
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved → {OUT_CSV}")

def mimic_cxr_embedding_FLP_multimodel():
    CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))

    BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_FILE_DIR))

    IMAGES_DIR = os.path.join(BASE_DIR, "mimiciv", "cxr")

    all_files = os.listdir(IMAGES_DIR)
    img_files = [f for f in all_files if f.lower().endswith(('.jpg','.jpeg','.png'))]

    dicom_ids = [f.rsplit('.', 1)[0] for f in img_files]

    meta = pd.DataFrame({"dicom_id": dicom_ids})


    meta["filepath"] = meta["dicom_id"].apply(lambda x: _find_correct_path(x, IMAGES_DIR))
    #meta["filepath"] = meta["dicom_id"].apply(lambda x: _find_correct_path(IMAGES_DIR,x))
    meta = meta.dropna(subset=["filepath"]).reset_index(drop=True)

    loader = DataLoader(CXRDataset2(meta, transform), batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Total valid images found: {len(meta)}")

    MODELS = ["densenet121", "resnet101", "efficientnet_b0"]

    OUTPUT_DIR = os.path.join(BASE_DIR, "data", "cohort")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for model_name in MODELS:
        print(f"\n🔹 Running embeddings for: {model_name}")

        feature_extractor, dim = build_feature_extractor(IMG_DIR, model_name)

        emb_list = [None] * len(meta)

        with torch.no_grad():
            for x, idxs in tqdm(loader, desc=f"Embedding: {model_name}"):
                x = x.to(device)
                f = feature_extractor(x)                     # (B, dim)
                f = torch.nn.functional.normalize(f, p=2, dim=1)

                for j, irow in enumerate(idxs.tolist()):
                    emb_list[irow] = f[j].cpu().numpy().tolist()

        # Save output CSV
        out_df = meta.copy()
        out_df["embedding"] = emb_list

        OUT_FILE = os.path.join(OUTPUT_DIR, f"cxr_embeddings_{model_name}.csv")
        out_df.to_csv(OUT_FILE, index=False)

        print(f"✔ Saved embeddings → {OUT_FILE}")
    
    