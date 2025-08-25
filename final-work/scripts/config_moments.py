# Импорт библиотек
import random
import numpy as np
import torch

# ------------------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

base_cfg = TrainConfig(
    model_name="DeepPavlov/rubert-base-cased-conversational",
    text_col="text",
    label_col="label",
    max_length=384,
    batch_size=16,
    epochs=4,
    lr=2e-5,
    weight_decay=0.01,
    warmup_prop=0.1,
    dropout=0.2,
    grad_clip_norm=1.0,
    use_amp=True,
    use_lora=True,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    loss_type="bce",
    target_precision=0.90,
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=42,
    num_workers=0,
    pin_memory=True,
    persistent_workers=False,
    prefetch_factor=None,
)

# base_cfg = TrainConfig(
#     model_name="sberbank-ai/ruRoberta-large",
#     text_col="text",
#     label_col="label",
#     max_length=512,
#     batch_size=8,
#     epochs=3,
#     lr=1e-5,
#     weight_decay=0.01,
#     warmup_prop=0.06,
#     dropout=0.2,
#     grad_clip_norm=1.0,
#     use_amp=True,
#     use_lora=True,
#     lora_r=8,
#     lora_alpha=16,
#     lora_dropout=0.1,
#     loss_type="bce",
#     target_precision=0.90,
#     device="cuda" if torch.cuda.is_available() else "cpu",
#     seed=42,
#     num_workers=0,
#     pin_memory=True,
#     persistent_workers=False,
#     prefetch_factor=None,
# )

# base_cfg = TrainConfig(
#     model_name="sberbank-ai/ruRoberta-large",
#     text_col="text",
#     label_col="label",
#     max_length=384,
#     batch_size=6,
#     epochs=3,
#     lr=1e-5,
#     weight_decay=0.01,
#     warmup_prop=0.06,
#     dropout=0.2,
#     grad_clip_norm=1.0,
#     use_amp=True,
#     use_lora=True,
#     lora_r=8,
#     lora_alpha=16,
#     lora_dropout=0.1,
#     loss_type="bce",
#     focal_gamma=2.0,
#     focal_alpha=0.25,
#     target_precision=0.90,
#     device="cuda" if torch.cuda.is_available() else "cpu",
#     seed=42,
#     num_workers=0,
#     pin_memory=True,
#     persistent_workers=False,
#     prefetch_factor=None,
#     grad_accum_steps=2
# )