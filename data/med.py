import os
import numpy as np
import tiktoken
from datasets import load_dataset

def process_chat_data(train_split=0.9):
    """
    Process chat data from the AI medical chatbot dataset.
    Combines the text in a structured format and tokenizes it.
    """
    # Load the dataset
    ds = load_dataset("ruslanmv/ai-medical-chatbot")
    
    # Convert dataset to formatted text
    def format_conversation(example):
        return f"Description: {example['description']}\nPatient: {example['patient']}\nDoctor: {example['doctor']}\n---\n"
    
    # Create the full text corpus
    # Assuming the main split we want to use is 'train'
    data = ''.join(map(format_conversation, ds['train']))
    
    # Split into train and validation sets
    n = len(data)
    train_data = data[:int(n*train_split)]
    val_data = data[int(n*train_split):]
    
    # Encode with tiktoken gpt2 bpe
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")
    
    # Export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    
    # Save to current directory
    train_ids.tofile('train.bin')
    val_ids.tofile('val.bin')
    
    return len(train_ids), len(val_ids)

if __name__ == "__main__":
    print("Loading and processing AI medical chatbot dataset...")
    
    try:
        train_tokens, val_tokens = process_chat_data()
        print(f"\nProcessing complete:")
        print(f"train.bin has {train_tokens:,} tokens")
        print(f"val.bin has {val_tokens:,} tokens")
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
