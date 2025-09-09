#script_dir = os.path.dirname(os.path.abspath(__file__))
#csv_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'MIMIC-IV-Data-Pipeline-main', 'mimiciv', 'notes'))
import os
import sys
import pandas as pd
import json
from tqdm import tqdm
sys.path.append(os.path.abspath("../utils"))
import icd_search
from icd_search import *
import re
import mimiciv_text_sectionizer
from mimiciv_text_sectionizer import *

import combination_util
from combination_util import *

SECTION_PATTERNS = [
    ("Alcohol",               r"(?:alcohol use|drinks?\s+alcohol|history of alcohol use|alcohol consumption)"),
    ("Benzodiazepine",        r"(?:benzodiazepines?|klonopin|valium|xanax|ativan)"),
    ("Opioid",                r"(?:prescription\s+opioid(?:s)?|opioid\s+use|use of (?:codeine|fentanyl|hydrocodone|methadone|oxycodone))"),
    
    ("Bipolar Disorder",      r"(?:bipolar\s*disorder|bpd)"),
    ("Schizophrenia",         r"(?:schizophrenia)"),
    ("Dementia",              r"(?:dementia)"),
    ("ADHD",                  r"(?:add|adhd|attention\s*deficit\s*hyperactive\s*disorder|attention\s*deficit\s*disorder)")
]


def get_csv(sec, cohort):
    '''
    Open .csv file from what section the user chose.

    Arg:
        sec (str) - section the user chose
    Returns:
        pandas Dataframe - the opened .csv file
    '''
    if isinstance(cohort, pd.DataFrame):
        if not cohort.empty:
            try:
                return cohort 
            except Exception:
                raise FileNotFoundError('No cohort DataFrame provided.')
        else:
            raise FileNotFoundError('DataFrame is empty')
    else:
        csv_path = os.path.join(base_dir, 'mimiciv', 'notes')
        
        try:
            return pd.read_csv(csv_path + '/' + sec.lower() + '.csv.gz', compression='gzip', header=0, index_col=None)
        except Exception as e:
            raise FileNotFoundError(f'CSV file not found: {csv_path}/{sec.lower()}.csv.gz\n\n{e}')

def section_splitting(df, sec):
    #using text_sectionizer created by Farzana
    print('Section splitting...')
    return mimiciv_text_sectionizer.extract_data_sectionizer(df, sec), 'SUMMARY PLACEHOLDER'
    

def extract_data(sec, pred, cohort):

    print('===========MIMIC-IV Notes============')
    df = get_csv(sec, cohort)
    
    summary = f'EXTRACTING FOR: {sec.upper()}, {pred.upper()}'
    
    f_df, sub_summary = prediction(pred, df, sec)
    summary += '\n' + sub_summary

    f_df.to_csv(base_dir+"/data/cohort/mimiciv_notes_cohort.csv.gz", index=False, compression='gzip')
    print("[ COHORT SUCCESSFULLY SAVED ]")

    with open(base_dir+'/data/cohort/notes_cohort_summary.txt', "w") as f:
        f.write(summary)

    print("[ SUMMARY SUCCESSFULLY SAVED ]")
    print(summary)

    return f_df

    

def prediction(pred, df, sec):
    
    match pred:
        case 'Entity':
            return entity_detection(df)

        case 'Context':
            return context_detection(df)  

        case 'Section':
            return section_splitting(df, sec)
    


            





    