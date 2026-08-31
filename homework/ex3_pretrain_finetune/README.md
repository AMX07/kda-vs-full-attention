# EX3: Pretraining then Finetuning

Goal: pretrain on a large corpus, then finetune on Tiny Shakespeare with fewer steps and a lower learning rate.

## Plan

- Pick a dataset large enough that train and validation loss stay close.
- Save the pretrained checkpoint in `models/`.
- Finetune from that checkpoint on Tiny Shakespeare.
- Compare against training Tiny Shakespeare from scratch.

## Results

Track validation loss, training time, checkpoint path, and sample generations.
