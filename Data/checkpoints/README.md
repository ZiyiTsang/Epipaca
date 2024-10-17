---
license: apache-2.0
library_name: peft
tags:
- trl
- sft
- generated_from_trainer
base_model: hfl/llama-3-chinese-8b-instruct
model-index:
- name: checkpoints
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# checkpoints

This model is a fine-tuned version of [hfl/llama-3-chinese-8b-instruct](https://huggingface.co/hfl/llama-3-chinese-8b-instruct) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.6416

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 2
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.03
- num_epochs: 2

### Training results



### Framework versions

- PEFT 0.11.1
- Transformers 4.40.2
- Pytorch 2.2.1
- Datasets 2.19.1
- Tokenizers 0.19.1