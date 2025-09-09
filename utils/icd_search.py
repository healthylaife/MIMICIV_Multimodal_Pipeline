import os, re
import pandas as pd
from tqdm import tqdm
import ahocorasick
from collections import defaultdict

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def normalize(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()

def _parse_versions(ver) -> set[int]:
    """accepts 9/10, 'v9', 'icd10', 'Both', ['v9','v10'] → {9,10} etc."""
    if isinstance(ver, (list, tuple, set)):
        out = set()
        for v in ver:
            out |= _parse_versions(v)
        return out
    v = str(ver).lower().strip()
    if v in {"both", "all"}:
        return {9, 10}
    v = v.replace("icd", "").replace("v", "")
    if v in {"9", "10"}:
        return {int(v)}
    raise ValueError(f"Unknown ICD version: {ver}")

STOP = {"unspecified","without","with","and","or","of","due","to","type","primary","secondary"}

def title_variants(long_title: str) -> set[str]:
    """cheap auto-variants from long title (no manual curation)."""
    t0 = normalize(long_title)
    t = t0.split(",")[0]
    toks = [w for w in t.split() if w not in STOP]
    variants = set()
    if t0: variants.add(t0)                    
    if toks: variants.add(" ".join(toks))      
    if len(toks) >= 2: variants.add(" ".join(toks[:2])) 
    return {v for v in variants if len(v) >= 3}

def _ensure_code_list(arg):
    if isinstance(arg, (list, tuple, set)):
        return [str(x).strip() for x in arg if str(x).strip()]
    if isinstance(arg, str):
        return [p.strip() for p in arg.split(",") if p.strip()]
    return [str(arg).strip()]

def _get_note_info(note_df, subject_id, hadm_id, col):
    match = note_df.loc[(note_df['subject_id'] == subject_id) & (note_df['hadm_id'] == hadm_id), col]
    note_value = match.iloc[0] if not match.empty else None
    return note_value

def search_icd(op1, modality, ver, args, df_name, use_variants=True):
    """
    df_name: notes csv stem (e.g. 'discharge' -> loads .../notes/discharge.csv.gz)
    ver: 9 / 10 / 'v9' / 'v10' / 'Both' / [9,10]
    args: list (or comma string) of ICD codes to search (['F0151','A431','G002'])
    """
    print('===========ICD SEARCH============')
    if op1:
        list_of_subjects = []
        icd_code_subject = pd.read_csv(os.path.join(base_dir, 'utils', 'mappings', 'diagnoses_icd.csv.gz'), compression='gzip')
        notes_df = pd.read_csv(os.path.join(base_dir, 'mimiciv', 'notes', f'{df_name}.csv.gz'), compression='gzip')
        for __, row in icd_code_subject.iterrows():
            if row['icd_code'] in args:
                subject_id = row.get('subject_id')
                hadm_id = row.get('hadm_id')
                list_of_subjects.append({
                    'note_id'    : _get_note_info(notes_df, subject_id, hadm_id, 'note_id'),
                    'subject_id' : subject_id,
                    'hadm_id'    : hadm_id,
                    'note_type'  : _get_note_info(notes_df, subject_id, hadm_id, 'note_type'),
                    'note_seq'   : _get_note_info(notes_df, subject_id, hadm_id, 'note_seq'),
                    'charttime'  : _get_note_info(notes_df, subject_id, hadm_id, 'charttime'),
                    'storetime'  : _get_note_info(notes_df, subject_id, hadm_id, 'storetime'),
                    'text'       : _get_note_info(notes_df, subject_id, hadm_id, 'text'),
                    'icd_code'   : row.get('icd_code'),
                    'icd_version': row.get('icd_version')
                    })
        f = pd.DataFrame(list_of_subjects)
        print(f)
        f.to_csv(os.path.join(base_dir, 'data', 'cohort', 'mimiciv_notes_cohort.csv.gz'), compression='gzip')
        return f
        
                    
                
        
    codes = _ensure_code_list(args)
    vers = _parse_versions(ver)
    print(f"Filtering by ICD {sorted(vers)}: {codes}")

    match modality:
        case 'notes':
            notes = pd.read_csv(os.path.join(base_dir, "mimiciv", "notes", f"{df_name}.csv.gz"), compression='gzip')
            data_sheet = notes
            print('Filtering ICD for Notes...')
        case 'cxr':
            cxr = pd.read_csv(os.path.join(base_dir, "utils", "mappings", "mapped_cxr_studies.csv.gz"), compression='gzip')
            data_sheet = cxr
            print('Filtering ICD for CXR Study notes..')
                
            

    icd = pd.read_csv(os.path.join(base_dir, "utils", "mappings", "icd_code_map.csv.gz"))


    icd = icd[icd["icd_version"].isin(vers)]
    if codes:
        icd = icd[icd["icd_code"].isin(codes)]
    if icd.empty:
        print("no icd rows after filtering (check codes/version)")
    icd = icd[["icd_code","icd_version","long_title"]].drop_duplicates().copy()

    icd['norm_title'] = icd['long_title'].map(normalize)

    term2payloads = defaultdict(set)
    for _, r in icd.iterrows():
        term = r["norm_title"]                  
        if term:
            term2payloads[term].add((
                r["icd_code"],                    
                r["long_title"],                  
                r["icd_version"],                  
            ))


    
    A = ahocorasick.Automaton()
    for term, payloads in term2payloads.items():
        A.add_word(term, tuple(payloads))
    A.make_automaton()


    
    rows = []
    for _, n in tqdm(data_sheet.iterrows(), total=len(data_sheet)):
        text_norm = normalize(n["text"]) if modality == "notes" else normalize(n["study"])
        seen = set()  
        for _, payloads in A.iter(text_norm):
            if payloads and not isinstance(payloads[0], (tuple, list)):
                payloads = (payloads,)
            for payload in payloads:
                if len(payload) != 3:
                    continue 
                code, title, ver_ = payload
                key = (code, ver_)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "note_id":   n.get("note_id"),
                    "subject_id": n["subject_id"],
                    "hadm_id":    n["hadm_id"],
                    "note_type":  n.get("note_type"),
                    "note_seq":   n.get("note_seq"),
                    "charttime":  n.get("charttime"),
                    "storetime":  n.get("storetime"),
                    "text": n.get("text"),
                    "icd_code":   code,
                    "icd_title":  title,
                    "icd_version": ver_,
                })

    matches_df = pd.DataFrame(rows)
    if not matches_df.empty:
        matches_df = matches_df.drop_duplicates(["note_id","icd_code","icd_version"])

        per_note_df = (
            matches_df
            .groupby(["note_id","subject_id","hadm_id","note_type","note_seq","charttime","storetime", "text"], dropna=False)
            .agg(
                icd_codes=("icd_code", lambda s: sorted(set(s))),
                icd_titles=("icd_title", lambda s: sorted(set(s))),
                icd_versions=("icd_version", lambda s: sorted(set(s))),
                n_codes=("icd_code", "nunique"),
            ).reset_index()
        )
    else:
        per_note_df = pd.DataFrame(columns=[
            "note_id","subject_id","hadm_id","note_type","note_seq","charttime","storetime", "text",
            "icd_codes","icd_titles","icd_versions","n_codes"
        ])


    out_dir = os.path.join(base_dir, "data", "cohort")
    os.makedirs(out_dir, exist_ok=True)
    vtag = "-".join([f"v{x}" for x in sorted(vers)])
    per_note_path = os.path.join(out_dir, f"sorted_icd_{df_name}_{vtag}.csv.gz")
    matches_path  = os.path.join(out_dir, f"matches_icd_{df_name}_{vtag}.csv.gz")
    per_note_df.to_csv(per_note_path, index=False, compression="gzip")
    matches_df.to_csv(matches_path,  index=False, compression="gzip")

    print(f"saved:\n  {per_note_path}\n  {matches_path}")
    summary = f'SIZE OF CSV {matches_df.size}'
    for arg in args:
        summary += f'\nNUM OF {arg} : {matches_df[matches_df["icd_code"] == arg].size}'
    print(summary)
    return matches_df

def diagnoses_search(df, icd):
    print(f'DIAGNOSES SEARCH | {df} | {icd}')
    modality_df = pd.read_csv(f'data/cohort/mimiciv_' + df + '_cohort.csv.gz', compression='gzip')
    icd_code_subject = pd.read_csv(os.path.join(base_dir, 'utils', 'mappings', 'diagnoses_icd.csv.gz'), compression='gzip')


    filtered_diagnoses = icd_code_subject[icd_code_subject['icd_code'].isin(icd)]


    matching_subjects = filtered_diagnoses['subject_id'].unique()


    filtered_df = modality_df[modality_df['subject_id'].isin(matching_subjects)]

    filtered_df.to_csv(f'data/cohort/mimiciv_{df}_cohort.csv.gz', compression='gzip')

    
    

        
        
        

    

