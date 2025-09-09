import os
import pandas as pd
from tqdm import tqdm
import wfdb
from pathlib import Path
import numpy as np
import torch 


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ecg_path = os.path.join(root_dir, 'mimiciv', 'ecg')
ecg_record_path = os.path.join(root_dir, 'utils', 'mappings')
#recursively find the root and append on to the record_list path


mapping_df = pd.read_csv(os.path.join(ecg_record_path, 'ecg_record_list.csv.gz'), compression='gzip')

def _turn_to_tensor(record_name):
    print(f'RECORD NAME: {record_name}')
    record_name = os.path.splitext(record_name)[0]
    record = wfdb.rdrecord(record_name)
    signal_data = record.p_signal
    ecg_tensor = torch.tensor(signal_data, dtype=torch.float32)
    return ecg_tensor

def _get_note_info(note_df, file_id, col):
    match = note_df[note_df['file_name'] == int(file_id)]
    note_match = match[col].iloc[0]
    print(f'MATCHING: note-id : {file_id}, column: {col}, match: {note_match}')
    note_value = match.iloc[0] if not match.empty else None
    return note_value

def _recursive_search(directory):
    data = []
    for file in Path(directory).rglob('*'):
        if file.is_file():
            data.append(file)
    return data


def _find_image_path(image_name):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if os.path.splitext(file)[0] == image_name:
                return os.path.join(root, file)

def extract_data(op, tensor_op):
    print('===========MIMIC-IV ECG============')
    records = []

    ecg_dir_list = os.listdir(os.path.join(root_dir, 'mimiciv', 'ecg'))
    ecg_dir = [d for d in ecg_dir_list if d.endswith('.dat')]
    directories = [d for d in ecg_dir_list if d not in ecg_dir]
    for d in directories:
        ecg_dir.extend(_recursive_search(d))

    print(f'ECG_DIR: {ecg_dir}')
    
    for file in tqdm(ecg_dir, total=len(ecg_dir)):
        file_name = os.path.splitext(file)[0]
        print(f'FILE NAME: {file_name}')
        file_path = _get_note_info(mapping_df, file_name, 'path')
        if op == 'Yes':
            file_path = _find_image_path(file_name)
        print(f'FILE_PATH: {file_path}')
        if tensor_op:
            records.append({
                'subject_id': _get_note_info(mapping_df, file_name, 'subject_id'),
                'study_id': _get_note_info(mapping_df, file_name, 'study_id'),
                'path': file_path,
                'ecg_tensor' : _turn_to_tensor(file_path)
            })
        else:      
            records.append({
                'subject_id': _get_note_info(mapping_df, file_name, 'subject_id'),
                'study_id': _get_note_info(mapping_df, file_name, 'study_id'),
                'path': file_path
            })
                    
    df = pd.DataFrame(records)
    print(f'SIZE: {df.size}')
    df.to_csv(os.path.join(root_dir, 'data', 'cohort', 'mimiciv_ecg_cohort.csv.gz'))
    print('[SAVED COHORT]')

    
    
    
    



    

    
    
    
    
