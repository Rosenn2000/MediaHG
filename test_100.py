import datasets
import random
import pandas as pd
from datasets import load_dataset
import warnings
from pathlib import Path
from typing import List, Tuple, Union
import torch
from rouge_chinese import Rouge
import jieba
import fire
from torch import nn
from datasets import Dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tokenizers_pegasus import PegasusTokenizer
from transformers import PreTrainedModel,DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.utils import logging
from transformers import PegasusForConditionalGeneration
from transformers.utils import logging
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from datasets import dataset_dict
import datasets

# model_checkpoint = "facebook/bart-large-cnn"
import gc


def preprocess_function(examples):
    prefix=""
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
def rouge_scores(
    summary, target_summary=None, token_budget=None,
):

    score = {}
    
    if target_summary is not None:
        summary = ' '.join(jieba.cut(summary))
        target_summary  = ' '.join(jieba.cut(target_summary))
        # rouge = rouge_score(summary, target_summary, rouge_ngrams=rouge_ngrams)
        rouge = Rouge()
        rouge_scores = rouge.get_scores(summary, target_summary)
        
    return rouge_scores

def BLEU_score(result,title):
    reference=list(jieba.cut_for_search(title))
    candidate=list(jieba.cut_for_search(result))
    score1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    score2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
    score3 =sentence_bleu(reference,candidate,weights=(0.33, 0.33, 0.33, 0))
    score4 = sentence_bleu(reference,candidate,weights=(0.25, 0.25, 0.25, 0.25))
    return score1,score2,score3,score4
def generate_summary(test_samples, model):
    inputs = tokenizer(
        test_samples,
        padding="max_length",
        truncation=True,
        max_length=max_input_length,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    outputs = model.generate(input_ids, attention_mask=attention_mask)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return output_str

def find_summary(dataloader,model):
    cnt=0
    rouge11,rouge22,rougeLsum1=0,0,0
    score11,score22,score33,score44=0,0,0,0
    for batch,batch_data in tqdm((enumerate(dataloader))):
        # print(batch_data)
        output=get_document_views(str(batch_data['content']))
        summary_views=get_summary_views(output,model)
        # print(output)
        title=''.join(batch_data['title']) 
        result=''.join(find_best_summary(summary_views,title))
        if result!='':
            score=rouge_scores(result,title)[0]
            rouge11 += score["rouge-1"]['f']
            rouge22 += score["rouge-2"]['f']
            rougeLsum1 += score["rouge-l"]['f']
            score1,score2,score3,score4=BLEU_score(result,title)
            score11 +=score1
            score22 +=score2
            cnt+=1
        else:
            continue
        
        # print(cnt)
        gc.collect()
    rouge1 = rouge11 / cnt
    rouge2 = rouge22 / cnt
    rougeLsum = rougeLsum1 / cnt
    score111=score11/cnt
    score222=score22/cnt

    print("ranking rouge1: %.6f, rouge2: %.6f, rougeL: %.6f"%(rouge1, rouge2, rougeLsum))
    print("ranking_BLEU1:%.6f,BLEU2:%.6f"%(score111,score222))

class xhsData(Dataset):
    def __init__(self,data_file): 
        self.data_dir=data_file
        self._data=self.preprocess(self.data_dir)
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
        return len(self._data)
    def __getitem__(self,idx):
        return self._data[idx]
    

if __name__ == "__main__":
    logger = logging.get_logger(__name__)
    torch.cuda.set_device(3)
    device=torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    model_checkpoint="IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese"

    tokenizer=PegasusTokenizer.from_pretrained(model_checkpoint)
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    model = PegasusForConditionalGeneration.from_pretrained("./models/pegasus_chinese")
    model = model.cuda()  
    max_input_length = 512 # input, source text
    max_target_length = 128 # summary, target text
    xhsdata=xhsData('6000.csv')
    test_data=DataLoader(xhsdata,batch_size=1,shuffle=True)
    if hasattr(torch.cuda, 'empty_cache'):
	    torch.cuda.empty_cache()
    with torch.no_grad():
        find_summary(test_data,model)


    rouge1 = rouge11 / cnt
    rouge2 = rouge22 / cnt
    rougeLsum = rougeLsum1 / cnt
    score11=score11/cnt
    score22=score22/cnt
    # score33+=score33/cnt
    # score44+=score44/cnt
    print("ranking rouge1: %.6f, rouge2: %.6f, rougeL: %.6f"%(rouge1, rouge2, rougeLsum))
    print("ranking_BLEU1:%.6f,BLEU2:%.6f,BLEU3:%.6f,BLEU4:%.6f"%(score11,score22,score33,score44))




