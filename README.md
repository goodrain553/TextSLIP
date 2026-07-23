# TextSLIP
I will upload the dataset and checkpoint ASAP.

TextSLIP combines the standard CLIP image-text contrastive loss with an ESimCSE text-text contrastive loss. The text branch uses a momentum encoder and a queue of momentum text features.

This repository intentionally does not include downstream evaluation, linear probing, retrieval, or dataset preparation scripts from the source workspace.

## Repository Layout

- `src/open_clip/`: CLIP model, ViT configs, HuggingFace text encoder, tokenizer, transforms, pretrained loading, and losses.
- `src/training/`: training entrypoint, WebDataset loading, distributed helpers, scheduler, logger, and training loops.
- `run_configs_textslip.py`: standalone TextSLIP brain MRI training configuration.
- `scripts/train_brain.sh`: environment-variable based launcher for single-node or multi-node `torchrun`.

## Installation

The `requirements.txt` file was exported from the working `unimed-clip` environment and pinned to package versions from that environment.

Create a conda environment:

```bash
conda create -n textslip python=3.10 -y
conda activate textslip
```

Install dependencies with pip:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If your machine needs a specific CUDA build of PyTorch, install PyTorch first with the command from the official PyTorch selector, then run the requirements installation. For example:

```bash
conda create -n textslip python=3.10 -y
conda activate textslip

# Example only: choose the CUDA version that matches your machine.
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

## Data Format


Training uses WebDataset shards. Each sample is expected to include an image field and a text field compatible with `src/training/data.py`.

The default `b16_400m_textslip_brain` config uses placeholder brain MRI shard patterns. Set your real shard list with:

```bash
export TEXTSLIP_TRAIN_DATA='/path/to/dataset-{000001..000010}.tar::/path/to/other-{000001..000005}.tar'
export TEXTSLIP_TRAIN_NUM_SAMPLES=7334713
```

If you use multiple dataset components, set matching upsampling factors:

```bash
export TEXTSLIP_TRAIN_DATA_UPSAMPLING_FACTORS='1::1'
```

## Training

Set the pretrained UniMed-CLIP/MetaCLIP checkpoint and run:

```bash
export TEXTSLIP_PRETRAINED=/path/to/b16_400m.pt
export TEXTSLIP_NGPUS=6
export TEXTSLIP_LOG_DIR=./logs

bash scripts/train_brain.sh
```

Useful overrides:

```bash
export TEXTSLIP_BATCH_SIZE=256
export TEXTSLIP_EPOCHS=40
export TEXTSLIP_LR=0.0001
export TEXTSLIP_ESIMCSE_SCALE=0.5
```

The training script launches:

```bash
torchrun src/training/main.py b16_400m_textslip_brain <log-dir>/<run-name> <pretrained-checkpoint>
```

The old config name `b16_400m_brain` is also available as an alias.

## Config Notes

The main TextSLIP config enables:

- `engine = "train_one_epoch_textSLIP"`
- `loss = "ESimCSECLIPLoss"`
- `use_tslip = True`
- `momentum_coef = 0.999`
- `tokenizer_context_length = 256`
- `text_encoder_model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"`

Checkpoints save the model, optimizer, scaler, and `loss_state`, including the ESimCSE queue.

## GitHub Upload Notes

Before pushing, make sure no datasets, checkpoints, logs, or private cluster paths are committed. The included `.gitignore` excludes common large artifacts, but review `git status` before publishing.

## Acknowledgement

This code is derived from UniMed-CLIP/OpenCLIP training code and keeps the original license file in this repository.
