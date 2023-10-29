# MediaHG

## About
Codes for EMNLP 2023 accepted paper: MediaHG: Rethinking Eye-catchy Features in Social Media Headline Generation

## Abstract
An attractive blog headline on social media platforms can immediately grab readers and trigger more clicks. However, a good headline shall not only contract the main content but also be eye-catchy with domain platform features, which are decided by the website’s users and objectives. With effective headlines, bloggers can obtain more site traffic and profits, while readers can have easier access to topics of interest. In this paper, we propose a disentanglement-based headline generation model: MediaHG (Social Media Headline Generation), which can balance the content and contextual features. Specifically, we first devise a sample module for various document views and generate the corresponding headline candidates. Then, we incorporate contrastive learning and auxiliary multi-task to choose the best domain-suitable headline, according to the disentangled budgets. Besides, our separated processing gains more flexible adaptation for other headline generation tasks with special domain features. Our model is built from the content and headlines of 70k hot posts collected from REDBook, a Chinese social media platform for daily sharing. Experimental results with language metrics ROUGE and human evaluation show the improvement in the headline generation task for the platform.

## File Folder
models: contains the parameters of BART and PEGASUS for chinese datasets

bart_chinese.py contains the main structure of headline generation process based on BART.
pegasus.py contains the main structure of headline generation process based on PEGASUS.
makedata.py contains the codes of select data from our dataset.
test_100.py contains the codes of style extraction and the candidate selection process.

## Demo
To run the eye-catcht headline generation, see example below

```
python test_100.py
```
