import os
import sys
import pandas as pd

sys.path.append(os.path.abspath("../preprocessing"))
import notes_preproc
from notes_preproc import *
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_cohort = '../data/cohort'

def get_combined_features_hosp(diag_flag=True, 
                               proc_flag=True, 
                               med_flag=True, 
                               lab_flag=True):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    feature_path = os.path.abspath(os.path.join(script_dir, '..', 'data', 'features'))

    features = []

    
    if diag_flag:
        diag = pd.read_csv(os.path.join(feature_path, 'preproc_diag.csv.gz'), compression='gzip')
        diag = diag.rename(columns={'new_icd_code': 'feature'})
        diag['type'] = 'diagnosis'
        features.append(diag)
    
    if proc_flag:
        proc = pd.read_csv(os.path.join(feature_path, 'preproc_proc.csv.gz'), compression='gzip')
        proc = proc.rename(columns={'icd_code': 'feature'})
        proc['type'] = 'procedure'
        features.append(proc)

    if med_flag:
        med = pd.read_csv(os.path.join(feature_path,'preproc_med.csv.gz'), compression='gzip')
        med = med.rename(columns={'drug_name': 'feature'})
        med['type'] = 'medication'
        features.append(med)

    if lab_flag:
        labs = pd.read_csv(os.path.join(feature_path, 'preproc_labs.csv.gz'), compression='gzip')
        labs = labs.rename(columns={'itemid': 'feature'})
        labs['type'] = 'lab'
        features.append(labs)

    for feature in features:
        notes_df = notes_preproc.get_csv()
        notes_grouped = notes_df.groupby('subject_id')
        all_results = []
        for _, tab_row in tab_df.iterrows():
            subject_id = tab_row['subject_id']
            if subject_id not in notes_grouped.groups:
                continue
            subject_notes = notes_grouped.get_group(subject_id)
            results = severity_dt(subject_notes)
            all_results.append(results)
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame(columns=['subject_id', 'note_id', 'label', 'ent', 'is_negated', 'is_historical', 'severity', 'icd_code'])    
        
    
def add_cxr_image(df):
    for file in os.listdir(data_cohort):
        file_name = 'mimiciv_cxr_data'
        if os.path.splitext(file)[0] == filename:
            cxr_path = os.path.join(data_cohort, filename)

    cxr_df = pd.read_csv(cxr_path, compression='gzip')
    
    df['study_id'] = None
    df['image_id'] = None
    df['rotation'] = None
    for i, row in df.iterrows():
        cxr_df_subject = cxr_df[cxr_df['subject_id'] == row['subject_id'].loc()[0]]
        df['study_id'] = cxr_df_subject['study_id'].loc()[0]
        df['image_id'] = cxr_df_subject['image_id'].loc()[0]
        df['rotation'] = cxr_df_subject['rotation'].loc()[0]

        
def _get_csv(name):
    path = os.path.join('data', 'cohort', f'mimiciv_{name}_cohort.csv.gz')
    try:
        f = pd.read_csv(path)
        print(f'{name}: {f.shape} shape')
        return f
    except Exception as e:
        print(f'Error loading {name}: {e}')
        return None

def raw_combine_modalities(arg_list, remove_nan):
    first_df = _get_csv(arg_list[0])
    if first_df is None:
        raise ValueError(f"Failed to load first modality: {arg_list[0]}")
    
    # Use subject_id + hadm_id if available, else just subject_id
    if 'hadm_id' in first_df.columns:
        id_cols = ['subject_id', 'hadm_id']
    else:
        id_cols = ['subject_id']
    
    merged_df = first_df[id_cols].copy()
    summary = f'MERGED CSV\nNUM OF SUBJECTS: {len(merged_df)}'

    for i, name in enumerate(arg_list, start=1):
        df = _get_csv(name)
        if df is None:
            print(f"Skipping {name} due to load failure.")
            continue
        
        # Determine join keys for this modality
        if 'hadm_id' in df.columns:
            join_keys = ['subject_id', 'hadm_id']
        elif 'subject_id' in df.columns:
            join_keys = ['subject_id']
        else:
            print(f"Skipping {name} due to missing subject_id.")
            continue

        col_to_add = [col for col in df.columns if col not in join_keys]
        merged_df = pd.merge(merged_df, df, on=join_keys, how='outer')
        summary += f'\n{name}|COLUMNS ADDED: {col_to_add}'

    summary += f'\nSIZE OF DATAFRAME: {merged_df.size}'
    summary += f'\nNUMBER OF SUBJECTS: {merged_df["subject_id"].nunique()}'
    summary += f'\nNUMBER OF POSITIVE COUNTS: {len(merged_df[merged_df["label"] == 1])}'
    summary += f'\nNUMBER OF NEGATIVE COUNTS: {len(merged_df[merged_df["label"] == 0])}'
    print(summary)
    if remove_nan:
        merged_df = merged_df.dropna()
    merged_df.to_csv(os.path.join(base_dir, 'data', 'output', 'combined_modalities.csv.gz'), compression='gzip')
    return merged_df




            
    
        

    
        

    
    