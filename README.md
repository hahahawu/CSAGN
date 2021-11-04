# CSAGN

The official source code of EMNLP'21 paper
["CSAGN: Conversational Structure Aware Graph Network for Conversational Semantic Role Labeling"](https://arxiv.org/abs/2109.11541).

## Dataset

You can find the CSRL data at [CSRL_dataset](https://github.com/syxu828/CSRL_dataset). Download the data and put them
into `model_data`.

## Setup

### Run
`python train.py MODEL_NAME -batch_size 128 -bert_version BERT_MODEL -intra_loss -inter_loss`

### Evaluation

`python evaluator.py TEST_DATASET MODEL_NAME`
