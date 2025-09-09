import os
import sys
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if device == 'cuda':
    print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])



tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2).to(device)
ckpt = torch.load(os.path.join(current_dir, 'bert_mortality.pt'), map_location=device)
state = ckpt.get("model_state", ckpt)  # handle dict or raw state_dict
missing, unexpected = model.load_state_dict(state, strict=False)
if not missing:
    print('✔ All weights successfully overwrote bert model.')
else:
    print('✖ Weights did not succesfully overwrite bert model.')
model.load_state_dict(state, strict=False)
model.eval()


@torch.no_grad()
def mortality_percent(text, max_len=512):
    enc = tokenizer(text, truncation=True, max_length=max_len, return_tensors='pt').to(device)
    p = torch.softmax(model(**enc).logits, dim=-1)[0,1].item()
    return 100.0 * p

def fwd_on_embeds(inputs_embeds, attention_mask):
    return model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits[:,1]

def ig_explain(text, baseline='zeros', max_len=512, n_steps=64):
    model.eval()
    enc = tokenizer(text, truncation=True, max_length=max_len, return_tensors='pt').to(device)
    print("n_tokens:", enc["input_ids"].shape[1])
    input_ids, attn = enc['input_ids'], enc['attention_mask']
    embed = model.get_input_embeddings()
    inputs_embeds = embed(input_ids)

    if baseline == 'zeros':
        base = torch.zeros_like(inputs_embeds)
    elif baseline == 'pad':
        base = embed(torch.full_like(input_ids, tokenizer.pad_token_id))
    elif baseline == 'mask':
        if tokenizer.mask_token_id is None:
            raise ValueError('No mask tokens.')
        base = embed(torch.full_like(input_is, tokenizer.mask_token_id))
    else:
        raise ValueError('Baseline must be zeros|pad|mask.')

    ig = IntegratedGradients(fwd_on_embeds)
    attr = ig.attribute(inputs_embeds, baselines=base, additional_forward_args=(attn,), n_steps=n_steps)
    token_attr = attr.sum(dim=-1).squeeze(0)
    toks = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())

    pairs = [(t, a.item()) for t,a in zip(toks, token_attr)
             if t not in {tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token}]
    prob = mortality_percent(text, max_len)/100.0
    return prob, pairs[:20]

text_input = input('Give example text...')
baseline_input = input('Baseline [zeros, pad, mask]')
p, top = tqdm(ig_explain(text_input, baseline=baseline_input.lower(), max_len=512, n_steps=64))
print(f'prob class: {p*100:.1f}%')
for tok, sc in top: print(f'{tok:>15} {sc:+.4f}')

    








