import os
import sys
import shutil
import pandas as pd
import spacy
from tqdm import tqdm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.transforms import transforms
from pathlib import Path

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath("../utils"))
import icd_search
from icd_search import *


def _recursive_search(directory):
    data = []
    for file in Path(directory).rglob('*'):
        if file.is_file():
            data.append(file)
    return data


#Need to add terget rules
#ADDED TENSOR IMAGE
def _turn_to_tensor(image_file):
    image = Image.open(os.path.join('mimiciv', 'cxr', image_file))
    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(image)
    return tensor
    
def _find_image_path(image_name):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if os.path.splitext(file)[0] == image_name:
                return os.path.join(root, file)


def _open_maps():
    data_path = os.path.join(base_dir, 'utils', 'mappings')
    metadata = pd.read_csv(os.path.join(data_path, 'mimic-cxr-2.0.0-metadata.csv.gz'), compression='gzip')
    df = pd.read_csv(os.path.join(data_path, 'mapped_cxr_studies.csv.gz'), compression='gzip')

    return metadata, df
    
def open_images(tensor, rotation):
    if rotation:
        print(f'ROTATIONS: {rotation}')
        
    metadata, df = _open_maps()
    metadata_list = []

    cxr_dir_list = os.listdir(os.path.join(base_dir, 'mimiciv', 'cxr'))
    cxr_dir = [d for d in cxr_dir_list if d.endswith('.dcm') or d.endswith('.jpg')]
    directories = [d for d in cxr_dir_list if d not in cxr_dir]
    for d in directories:
        cxr_dir.extend(_recursive_search(d))
        

        


    for image in tqdm(cxr_dir, total=len(cxr_dir)):

        cxr_path = _find_image_path(os.path.splitext(image)[0])
        dicom_id = os.path.splitext(image)[0]
        type_id = os.path.splitext(image)[-1]

        image_metadata = metadata[metadata['dicom_id'] == dicom_id]

        filtered = image_metadata[image_metadata['ViewCodeSequence_CodeMeaning'].isin(rotation)]

        if filtered.empty:
            continue
                    
        rotation_value = filtered['ViewCodeSequence_CodeMeaning'].iloc[0]
        subject_id = filtered['subject_id'].iloc[0]
        study_id = filtered['study_id'].iloc[0]
        study_text = df['study'].iloc[0]
        if tensor:
            image_tensor = _turn_to_tensor(image)
            metadata_list.append([dicom_id, cxr_path, subject_id, study_id, rotation_value, study_text, image_tensor])

        else:
            metadata_list.append([dicom_id, cxr_path, subject_id, study_id, rotation_value, study_text])

    if tensor:
        f_df = pd.DataFrame(metadata_list, columns=['image_id', 'image_path', 'subject_id', 'study_id', 'rotation', 'study_text', 'image_tensor'])
    else:
        f_df = pd.DataFrame(metadata_list, columns=['image_id', 'image_path', 'subject_id', 'study_id', 'rotation', 'study_text'])
    cohort_path = os.path.join(base_dir, 'data', 'cohort')

        
        
    f_df.to_csv(os.path.join(cohort_path, 'mimiciv_cxr_cohort.csv.gz'), compression='gzip')
    print('[IMAGE COHORT SUCCESSFULLY SAVED]')

    rotation_counts = f_df['rotation'].value_counts()

    summary = '\n'.join([
                'SUMMARY',
                f'SHAPE OF COHORT: {f_df.shape}',
                f'TYPE OF IMAGES: {type_id}',
                f"NUM OF ROTATION lateral: {len(f_df[f_df['rotation'] == 'lateral'])}",
                f"NUM OF ROTATION left lateral: {len(f_df[f_df['rotation'] == 'left lateral'])}",
                f"NUM OF ROTATION anterior-postero: {len(f_df[f_df['rotation'] == 'anterior-postero'])}",
                f"NUM OF ROTATION postero-anterior: {len(f_df[f_df['rotation'] == 'postero-anterior'])}"]
        )
    base_path = os.path.join(base_dir, 'data', 'cohort', 'mimiciv_cxr_summary.txt')
    version = 1
    final_path = base_path
        
    while os.path.exists(final_path):
        final_path = os.path.join(
            cohort_path, f"mimiciv_cxr_summary_{version}.txt"
        )
        version += 1
        
    with open(final_path, 'w') as file:
        file.write(summary)
        
    print(f'{summary}\n[IMAGE SUMMARY SUCCESSFULLY SAVED]')
    return summary

            


    



    

    
                
                


    

        
            
            

    
    
    