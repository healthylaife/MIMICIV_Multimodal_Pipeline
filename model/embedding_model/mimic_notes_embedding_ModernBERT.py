import pandas as pd
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

MODEL_NAME = "answerdotai/ModernBERT-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)

# --------------- INTERNAL FUNCTIONS -----------------------------
def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def _embed_text(text, max_len=512, stride=64):
    #usse the long texts and chunking them --> caluculating the avg of the embeddings
    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_len,
        stride=stride,
        return_overflowing_tokens=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    chunk_embs = []
    for i in range(len(enc["input_ids"])):
        input_ids = enc["input_ids"][i].unsqueeze(0).to(device)
        attn_mask = enc["attention_mask"][i].unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask)
            emb = _mean_pool(out.last_hidden_state, attn_mask)  # (1,768)
            chunk_embs.append(emb[0])

    note_emb = torch.stack(chunk_embs).mean(dim=0)  # average chunks
    note_emb = torch.nn.functional.normalize(note_emb, p=2, dim=0)  # normalize
    return note_emb.cpu().tolist()

# ----------------------OUPUT FUNCTION ---------------------------
def mimic_notes_embedding_ModernBERT(mimic_note_type):

    CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))

    BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_FILE_DIR))
    df = pd.read_csv(f"{BASE_DIR}/mimiciv/notes/{mimic_note_type}.csv.gz", compression='gzip',nrows=200)

    TEXT_COL = "text"
    NOTE_ID_COL = "note_id"
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

    #using the model https://huggingface.co/answerdotai/ModernBERT-base
    model.eval()

    all_embs = []
    for text in tqdm(df[TEXT_COL].tolist(), desc="Embedding notes"):
        all_embs.append(_embed_text(text))

    df["text_embed"] = all_embs

    out_path = f"{BASE_DIR}/data/cohort/notes_{mimic_note_type}_with_modernbert_embedding.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved embeddings → {out_path}")
    print(df[[NOTE_ID_COL, "text_embed"]].head(2))

    
    