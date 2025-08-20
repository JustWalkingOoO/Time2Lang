<<<<<<< HEAD
## Time2Lang

![Logo](.\img\Logo.png)

Here is the code for the paper "Time2Lang: A Frequent-Pattern-Driven Translation Mechanism for Zero-Shot Time Series Forecasting with LLMs".

## Requirements

- Python 3.10.16
- Detail in `requirements.txt`

## Directory Structure

1. pretraining
   - PrefixSpan: Frequent sequence mining
   - VQVAE: Training VQVAE for discretization

2. inference
   - Code for the experimental inference phase

## Dataset

![dataset](.\img\dataset.png)

You can get test data from https://github.com/JustWalkingOoO/Time2Lang/releases/tag/Resource, then place it to `./inference/`

## Pre-training Model

- TimeGPT: https://dashboard.nixtla.io/freetrial
- Chronos: https://huggingface.co/amazon/chronos-t5-large
- Moirai: https://huggingface.co/Salesforce/moirai-1.0-R-large
- TimesFM: https://huggingface.co/google/timesfm-2.0-500m-pytorch
- LLaMA2: https://huggingface.co/meta-llama/Llama-2-7b-hf
- ChatTime: https://huggingface.co/ChengsenWang/ChatTime-1-7B-Chat

## Running

#### Step 1

Install dependencies as `requirements.txt`

```shell
pip install -r requirements.txt
```

#### Step 2

Get trained VQVAE from https://github.com/JustWalkingOoO/Time2Lang/releases/tag/Resource or train by yourself. And then place it to `./inference/`

#### Step 3

Zero-shot Lab

```shell
python token2token-zero-shot.py
python token2time-zero-shot.py
```

Few-shot Lab

```shell
python token2token-zero-shot.py
python token2time-zero-shot.py
=======
## Time2Lang

![Logo](.\img\Logo.png)

Here is the code for the paper "Time2Lang: A Frequent-Pattern-Driven Translation Mechanism for Zero-Shot Time Series Forecasting with LLMs".

## Requirements

- Python 3.10.16
- Detail in `requirements.txt`

## Directory Structure

1. pretraining
   - PrefixSpan: Frequent sequence mining
   - VQVAE: Training VQVAE for discretization

2. inference
   - Code for the experimental inference phase

## Dataset

![dataset](.\img\dataset.png)

You can get test data from https://github.com/JustWalkingOoO/Time2Lang/releases/tag/Resource, then place it to `./inference/`

## Pre-training Model

- TimeGPT: https://dashboard.nixtla.io/freetrial
- Chronos: https://huggingface.co/amazon/chronos-t5-large
- Moirai: https://huggingface.co/Salesforce/moirai-1.0-R-large
- TimesFM: https://huggingface.co/google/timesfm-2.0-500m-pytorch
- LLaMA2: https://huggingface.co/meta-llama/Llama-2-7b-hf
- ChatTime: https://huggingface.co/ChengsenWang/ChatTime-1-7B-Chat

## Running

#### Step 1

Install dependencies as `requirements.txt`

```shell
pip install -r requirements.txt
```

#### Step 2

Get trained VQVAE from https://github.com/JustWalkingOoO/Time2Lang/releases/tag/Resource or train by yourself. And then place it to `./inference/`

#### Step 3

Zero-shot Lab

```shell
python token2token-zero-shot.py
python token2time-zero-shot.py
```

Few-shot Lab

```shell
python token2token-zero-shot.py
python token2time-zero-shot.py
>>>>>>> 8f5ca24 (Initial commit: add pretraining and inference modules)
```