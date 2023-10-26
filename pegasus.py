import datasets
import random
import pandas as pd
from datasets import load_dataset,dataset_dict
import warnings
from pathlib import Path
from typing import List, Tuple, Union
import torch
torch.cuda.set_device(3)
device=torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

import fire
from torch import nn

from tokenizers_pegasus import PegasusTokenizer
from transformers import PreTrainedModel,DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.utils import logging
from transformers import PegasusForConditionalGeneration
logger = logging.get_logger(__name__)
# model_checkpoint = "facebook/bart-large-cnn"

model_checkpoint="IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese"

tokenizer=PegasusTokenizer.from_pretrained(model_checkpoint)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model = PegasusForConditionalGeneration.from_pretrained(model_checkpoint)
# model=model.to(device)
dataset=load_dataset("csv", data_files="process.csv")

def flatten(example):
    return {
        "document": example["content"],
        "summary": example["title"],
        "id":"0"
    }
Dataset = dataset["train"].map(flatten, remove_columns=["title", "content"])

model = model.cuda()  # 在使用DistributedDataParallel之前，需要先将模型放到GPU上
# model = torch.nn.parallel.DistributedDataParallel(my_model, find_unused_parameters=True)
max_input_length = 512 # input, source text
max_target_length = 128 # summary, target text

def preprocess_function(examples):
    prefix=""
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


train_data_txt, validation_data_txt = Dataset.train_test_split(test_size=0.1).values()
train_data_txt, test_data_tex = train_data_txt.train_test_split(test_size=0.1).values()
# 装载数据
dd = datasets.DatasetDict({"train":train_data_txt,"validation": validation_data_txt,"test":test_data_tex }) 

raw_datasets = dd
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
batch_size = 4
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
args = Seq2SeqTrainingArguments(
    output_dir="results",
    num_train_epochs=1,  # demo
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=batch_size,  # demo
    per_device_eval_batch_size=batch_size,
    # learning_rate=3e-05,
    warmup_steps=500,
    weight_decay=0.1,
    label_smoothing_factor=0.1,
    predict_with_generate=True,
    logging_dir="logs",
    logging_steps=50,
    save_total_limit=3,
)
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(jieba.cut(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(jieba.cut(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

if hasattr(torch.cuda, 'empty_cache'):
	torch.cuda.empty_cache()
trainer.train()
model.save_pretrained("./models/pegasus_chinese")

