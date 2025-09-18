from __future__ import annotations
import os
import torch
import numpy as np
from lightning.pytorch import Trainer
from BLT.entropy import EntropyModel, EntropyConfig, EntropyWrapper, configure_optimizer
from BLT.dataset import TokenDataset
from huggingface_hub import hf_hub_download




# download the checkpoint for the model
hf_hub_download(repo_id="pt-sk/BLT_Entropy_Checkpoints",
                filename="entropy_ckpt_4.ckpt",
                repo_type="model",
                local_dir="/kaggle/working/")

# download the tokenized text
text_files = ["tokenized_text5.npy", "tokenized_text6.npy"]
for file in text_files:
    hf_hub_download(repo_id="pt-sk/Text_Bytes_Tokens",
                    filename=f"wikipedia_512_pretraining/{file}",
                    repo_type="dataset",
                    local_dir="/kaggle/working/")

# load the tokenized text
tokens = []
for file in text_files:
    tokens += np.load(f"/kaggle/working/wikipedia_512_pretraining/{file}", allow_pickle=True).tolist()
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
                                        batch_size=6, 
                                        shuffle=True,
                                        pin_memory=True,
                                        pin_memory_device='cuda',
                                        num_workers=os.cpu_count(),
                                        prefetch_factor=2)

# Initialize model wrapper
model_wrapper = EntropyWrapper.load_from_checkpoint("/kaggle/working/entropy_ckpt_4.ckpt", config=config, model=model)

# Initialize trainer
trainer = Trainer(max_epochs=1,
                  accelerator="cuda",
                  accumulate_grad_batches=8,
                  gradient_clip_val=1.0,
                  devices=1,
                  strategy="ddp")

# Train the model
trainer.fit(model_wrapper, dataloader)