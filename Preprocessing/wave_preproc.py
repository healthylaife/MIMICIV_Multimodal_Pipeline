import os
import pandas as pd
from tqdm import tqdm


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wave_path = os.path.join(root_dir, 'mimiciv', 'wave')
wave_record_path = os.path.join(root_dir, 'utils', 'mappings')
#recursively find the root and append on to the record_list path


mapping_df = pd.read_csv(os.path.join(wave_record_path, 'wave_record_list.csv.gz'), compression='gzip')

def _get_note_info(note_df,file_id, col):
    match = note_df.loc[(note_df['subject_id'] == file_id), col]
    note_value = match.iloc[0] if not match.empty else None
    return note_value

def _find_image_path(image_name):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if os.path.splitext(file)[0] == image_name:
                return os.path.join(root, file)

def extract_data(op):
    print('===========MIMIC-IV WAVEFORM============')
    records = []
    for file in tqdm(os.listdir(wave_path), total=len(os.listdir(wave_path))):
        file_name = os.path.splitext(file)[0]
        if op == 'Yes':
            file_path = _find_image_path(file_name)
        records.append({
            'subject_id': _get_note_info(mapping_df, file_name, 'subject_id'),
            'study_id': file_name,
            'wave_id': _get_note_info(mapping_df, file_name, 'wave_id'),
            'path': file_path
        })
                
    df = pd.DataFrame(records)
    print(f'SIZE: {df.size}')
    df.to_csv(os.path.join(root_dir, 'data', 'cohort', 'mimiciv_wave_cohort.csv.gz'))
    print('[SAVED COHORT]')