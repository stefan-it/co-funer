---
language: de
license: mit
tags:
- flair
- token-classification
- sequence-tagger-model
- hetzner
- hetzner-gex44
- hetzner-gpu
base_model: {{ base_model }}
widget:
- text: {{ widget_text }}
---

# Fine-tuned Flair Model on CO-Fun NER Dataset

This Flair model was fine-tuned on the
[CO-Fun](https://arxiv.org/abs/2403.15322) NER Dataset using {{ base_model_short }} as backbone LM.

## Dataset

The [Company Outsourcing in Fund Prospectuses (CO-Fun) dataset](https://arxiv.org/abs/2403.15322) consists of
948 sentences with 5,969 named entity annotations, including 2,340 Outsourced Services, 2,024 Companies, 1,594 Locations
and 11 Software annotations.

Overall, the following named entities are annotated:

* `Auslagerung` (engl. outsourcing)
* `Unternehmen` (engl. company)
* `Ort` (engl. location)
* `Software`

## Fine-Tuning

The latest [Flair version](https://github.com/flairNLP/flair/tree/42ea3f6854eba04387c38045f160c18bdaac07dc) is used for
fine-tuning.

A hyper-parameter search over the following parameters with 5 different seeds per configuration is performed:

* Batch Sizes: {{ batch_sizes }}
* Learning Rates: {{ learning_rates }}

More details can be found in this [repository](https://github.com/stefan-it/co-funer). All models are fine-tuned on a
[Hetzner GX44](https://www.hetzner.com/dedicated-rootserver/matrix-gpu/) with an NVIDIA RTX 4000.

## Results

A hyper-parameter search with 5 different seeds per configuration is performed and micro F1-score on development set
is reported:

{{ results }}

The result in bold shows the performance of the current viewed model.

Additionally, the Flair [training log](training.log) and [TensorBoard logs](../../tensorboard) are also uploaded to the model
hub.
