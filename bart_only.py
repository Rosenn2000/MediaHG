import torch
torch.cuda.set_device(1)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device=torch.device('cpu')
print(device)
import pandas as pd
import random
from itertools import combinations
import numpy as np

TokenModel = "bert-base-chinese"
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(TokenModel)
from tqdm import tqdm
import warnings
from pathlib import Path
from typing import List, Tuple, Union
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import fire
from torch import nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel
from transformers.utils import logging
from datasets import Dataset

from rouge_chinese import Rouge
import jieba

model=AutoModelForSeq2SeqLM.from_pretrained('./models/bart_chinese')
def generate_summary(test_samples, model):
    inputs = tokenizer(
        test_samples,
        padding="max_length",
        truncation=True,
        max_length=1024,
        return_tensors="pt",
        is_split_into_words=True
    )
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    with torch.no_grad:
        outputs = model.generate(input_ids, attention_mask=attention_mask,max_length=32,no_repeat_ngram_size=2,num_beams=4)
        output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return output_str
class xhsData(Dataset):
    def __init__(self,data_file): 
        
        self.data_dir=data_file
        self.data=self.preprocess(self.data_dir)
        
    def preprocess(self,data_dir):
        Data={}
        
        sv=pd.read_csv(data_dir)
        df=pd.DataFrame(sv)
        for idx,row in df.iterrows():
            title=row['title']
            
            content=row['content']
            Data[idx]={
                "title":title,
                "content":content
            }
        return Data
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]
data=xhsData('test_100.csv')
batch = next(iter(data))
print(batch)