# mm2023-conversation-stance

## Overview
Codes for the accepted paper in ICDM 2023: "Contextual Target-Specific Stance Detection on Twitter: New Dataset and Method". It includes the implementation of the ConMulAttn model and the necessary scripts for training and testing.

Dataset and model checkpoints can be download at this [link](https://drive.google.com/drive/folders/1oc7CVpJPo1M6x1JHVZaHaYaeMNk6tbwQ?usp=share_link). Our dataset, CTSDT, consists of 35,108 conversations and 1,707,564 tweets regarding the topic of COVID-19 vaccination on Twitter. This is the first dataset for the study of contextual target-specific stance detection on Twitter. We are also providing pre-trained models for ConMulAttn on CTSDT and other popular datasets.

## Dependencies
```
numpy>=1.22
pandas==1.1.5
scikit_learn==1.2.2
sentence_transformers==2.2.2
torch==1.10.2
tqdm==4.64.0
```

## Usage

To run our proposed methods:
```
python run.py
```

To run baselines:
```
python "baselines/run.py" --model %MODEL_NAME%
```
