"""Training configs for standalone TextSLIP experiments."""

import os
from dataclasses import dataclass

from configs import Config


BRAIN_TRAIN_DATA = "::".join(
    [
        "/path/to/WHM_im_webdataset/dataset-{000001..000008}.tar",
        "/path/to/ISLES_im_webdataset/dataset-{000001..000032}.tar",
        "/path/to/BraTS2023_glioma_webdataset/dataset-{000001..000005}.tar",
        "/path/to/BraTS-MEN_im/dataset-{000001..000008}.tar",
        "/path/to/BraTS-MEN_im_grade2/dataset-{000001..000003}.tar",
        "/path/to/MT_BraTS2013/dataset-{000001..000101}.tar",
        "/path/to/MT_CrossMoDA22/dataset-{000001..000002}.tar",
        "/path/to/MT_ISLES2016/dataset-{000001..000002}.tar",
        "/path/to/MT_ISLES2017/dataset-{000001..000002}.tar",
        "/path/to/MT_ISLES2022/dataset-{000001..000010}.tar",
        "/path/to/MT_ISLES_SISS/dataset-{000001..000012}.tar",
        "/path/to/MT_ISLES_SPES/dataset-{000001..000016}.tar",
        "/path/to/MT_LMSLS/dataset-{000001..000010}.tar",
        "/path/to/BraTS2015/dataset-{000001..000518}.tar",
        "/path/to/BraTS2018/dataset-{000001..000234}.tar",
        "/path/to/BraTS2019/dataset-{000001..000548}.tar",
        "/path/to/MT_CrossMoDA21/dataset-000001.tar",
        "/path/to/MT_BraTS2021/dataset-{000001..001173}.tar",
        "/path/to/MT_BraTS2020/dataset-{000001..000627}.tar",
        "/path/to/MT_BrainTumour/dataset-{000001..000829}.tar",
        "/path/to/MT_Brain_PTM/dataset-{000001..000010}.tar",
        "/path/to/MT_brats/dataset-{000001..000865}.tar",
        "/path/to/MT_brats24/dataset-{000001..001852}.tar",
        "/path/to/MT_brats_24/dataset-{000001..000372}.tar",
    ]
)

BRAIN_UPSAMPLING_FACTORS = "::".join(["1"] * 24)


@dataclass
class b16_400m_textslip_brain(Config):
    """ViT-B/16 TextSLIP training config for brain MRI WebDataset shards."""

    inmem = True
    engine = "train_one_epoch_textSLIP"
    eval_steps = 5000
    save_frequency = 2
    save_most_recent = True

    model = "ViT-B-16-quickgelu"
    name = "ViT-B-16-TextSLIP-Brain"
    force_quick_gelu = True
    grad_checkpointing = True
    text_encoder_model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    tokenizer_context_length = 256

    loss = "ESimCSECLIPLoss"
    use_tslip = True
    esimcse_scale = float(os.environ.get("TEXTSLIP_ESIMCSE_SCALE", "0.5"))
    esimcse_temperature = float(os.environ.get("TEXTSLIP_ESIMCSE_TEMPERATURE", "0.05"))
    esimcse_neg_size = int(os.environ.get("TEXTSLIP_ESIMCSE_NEG_SIZE", "128"))
    momentum_coef = float(os.environ.get("TEXTSLIP_MOMENTUM_COEF", "0.999"))
    text_aug_word_dup = False
    text_aug_dup_rate = 0.3

    train_data = os.environ.get("TEXTSLIP_TRAIN_DATA", BRAIN_TRAIN_DATA)
    train_num_samples = int(os.environ.get("TEXTSLIP_TRAIN_NUM_SAMPLES", "7334713"))
    train_data_upsampling_factors = os.environ.get(
        "TEXTSLIP_TRAIN_DATA_UPSAMPLING_FACTORS",
        BRAIN_UPSAMPLING_FACTORS,
    )
    dataset_resampled = True

    pretrained = os.environ.get("TEXTSLIP_PRETRAINED", "")
    logs = os.environ.get("TEXTSLIP_LOG_DIR", "./logs")
    report_to = os.environ.get("TEXTSLIP_REPORT_TO", "tensorboard")

    nodes = int(os.environ.get("TEXTSLIP_NODES", "1"))
    ngpus = int(os.environ.get("TEXTSLIP_NGPUS", "6"))
    workers = int(os.environ.get("TEXTSLIP_WORKERS", "4"))
    batch_size = int(os.environ.get("TEXTSLIP_BATCH_SIZE", "256"))
    epochs = int(os.environ.get("TEXTSLIP_EPOCHS", "40"))
    warmup = int(os.environ.get("TEXTSLIP_WARMUP", "5000"))
    lr = float(os.environ.get("TEXTSLIP_LR", "0.0001"))
    seed = int(os.environ.get("TEXTSLIP_SEED", "0"))

    local_loss = True
    gather_with_grad = True
    imagenet_val = None
    eval_freq = int(os.environ.get("TEXTSLIP_EVAL_FREQ", "999"))


# Backward-compatible alias for commands copied from the source workspace.
b16_400m_brain = b16_400m_textslip_brain
