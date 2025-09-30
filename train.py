from __future__ import annotations
import os
import torch
import numpy as np
from lightning.pytorch import Trainer
from BLT.entropy import EntropyModel, EntropyConfig, EntropyWrapper, configure_optimizer
from BLT.dataset import TokenDataset
from huggingface_hub import hf_hub_download



# Define constants
LOCAL_DIR = os.getcwd()
MODEL_REPO = "pt-sk/BLT_Entropy_Checkpoints"
DATASET_REPO = "pt-sk/Text_Bytes_Tokens"
DATASET_REPO_FOLDER = "wikipedia_512_pretraining"
CKPT_NAME = "entropy_ckpt_9.ckpt"
TEXT_FILES = ["tokenized_text12.npy", "tokenized_text13.npy"]


# download the checkpoint for the model
hf_hub_download(repo_id=MODEL_REPO,
                filename=CKPT_NAME,
                repo_type="model",
                local_dir=LOCAL_DIR)

# download the tokenized text
for file in TEXT_FILES:
    hf_hub_download(repo_id=DATASET_REPO,
                    filename=f"{DATASET_REPO_FOLDER}/{file}",
                    repo_type="dataset",
                    local_dir=LOCAL_DIR)

# load the tokenized text
tokens = []
for file in TEXT_FILES:
    tokens += np.load(f"{LOCAL_DIR}/{DATASET_REPO_FOLDER}/{file}", allow_pickle=True).tolist()
print(f"Loaded {len(tokens)} tokenized sequences.")

# Initialize model and config
config = EntropyConfig()
model = EntropyModel(config)

# count the number of parameters in the model
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters in the model: {num_params/1e6}M")

# initialize optimizer
optimizer = configure_optimizer(model)

# Create dataset and dataloader
dataset = TokenDataset(block_size=4096, input_ids=tokens)
dataloader = torch.utils.data.DataLoader(dataset, 
                                        batch_size=4, 
                                        shuffle=True,
                                        num_workers=os.cpu_count(),
                                        prefetch_factor=4)

# Initialize model wrapper
model_wrapper = EntropyWrapper.load_from_checkpoint(f"{LOCAL_DIR}/{CKPT_NAME}", config=config, model=model)

# Initialize trainer
trainer = Trainer(max_epochs=1,
                  accelerator="cuda",
                  accumulate_grad_batches=8,
                  gradient_clip_val=1.0,
                  devices=2,
                  strategy="ddp")

# Train the model
trainer.fit(model_wrapper, dataloader)