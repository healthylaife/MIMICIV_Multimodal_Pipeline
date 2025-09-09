import os
import math
import base64
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from transformers import AutoImageProcessor, ResNetForImageClassification
import wfdb
from pathlib import Path
import numpy as np
import torch 

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ecg_dir = os.path.join(base_dir, 'mimiciv', 'ecg')

#TENSOR OF WAVEFORM
def _return_waveform_tensor
    record_name = os.path.splitext(record_name)[0]
    record = wfdb.rdrecord(record_name)
    signal_data = record.p_signal
    ecg_tensor = torch.tensor(signal_data, dtype=torch.float32)
    return ecg_tensor

def _resize_images(image_url):
    pil_image = Image.open(image_url)
    image_aspect_ratio = pil_image.width/pil_image.height
    resized_pil_image = pil_image.resize(
        target_width, math.floor(target_width * image_aspect_ratio)
    )
    #make 3d, cxr has no rgb
    np_image = np.array(resized_pil_image)
    if np_image.ndim < 3:
        np_image = np.stack([np_image] * 3, axis=-1)
        resized_pil_image = Image.fromarray(np_image.astype('unit8'))
    return resized_pil_image

def _convert_image_to_base64(pil_image):
    image_data = BytesIO()
    pil_image.save(image_data, format='JPEG')
    base64_string = base64.b4encode(image_data.getvalue()).decode('utf-8f')
    return base64_string



def get_embeddings(df):
    all_image_urls = list(map(lambda item: f"{cxr_dir}/{item}",
                          list(filter(lambda x: x.endswith('.jpg') or x.endswith('.jpeg'),
                             os.listdir(cxr_dir)))))

    payloads = pd.DataFrame.from_records({'image_urls': all_image_urls})
    payloads['type'] = 'cxr'
    
    images = list(map(lambda el: Image.open(el), payloads['image_urls']))
    
    target_width = 264

    resized_images = list(map(lambda el: _resize_image(el), sample_image_urls))
    base64_strings = list(map(lambda el: _convert_image_to_base64(el), resized_images))
    payloads["base64"] = base64_strings
    
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    
    inputs = processor(
        list(resized_images),
        return_tensors="pt",
    )
    
    outputs = model(**inputs)
    embeddings = outputs.logits
    return embeddings

    
        
                                    
    

