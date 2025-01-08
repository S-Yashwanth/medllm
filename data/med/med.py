import os
import numpy as np
from datasets import load_dataset
import tiktoken

# Load the dataset
ds = load_dataset("ruslanmv/ai-medical-chatbot")

# Extract the columns from the dataset and prepare the data
data = ""
for row in ds['train']:
    description = row['Description']
    patient = row['Patient']
    doctor = row['Doctor']
    data += f"Description: {description}\nPatient: {patient}\nDoctor: {doctor}\n\n"

# Split data into training and validation sets
n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]

# Encode with tiktoken using GPT-2 byte pair encoding
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Set the output directory
output_dir = '/kaggle/working/medllm/data/shakespeare_char'
os.makedirs(output_dir, exist_ok=True)

# Save the tokenized data to binary files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(output_dir, 'train.bin'))
val_ids.tofile(os.path.join(output_dir, 'val.bin'))

print(f"train.bin and val.bin have been saved to {output_dir}")
