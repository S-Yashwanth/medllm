import os
import argparse
import torch
import tiktoken
from model import GPTConfig, GPT

def load_model(out_dir, device):
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)

    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']

    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model

def chatbot(out_dir, start, num_samples, max_new_tokens, temperature=0.7, top_k=50):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(out_dir, device)
    
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

    input_ids = encode(start)
    x = torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        for _ in range(num_samples):
            output_ids = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            response = decode(output_ids[0].tolist())
            print(f"Sample {_ + 1}: {response}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chatbot using a trained GPT model")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory where the trained model checkpoint is stored")
    parser.add_argument("--start", type=str, default="Hello", help="Initial text prompt for the chatbot")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of generated responses")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum new tokens to generate")
    
    args = parser.parse_args()
    
    chatbot(args.out_dir, args.start, args.num_samples, args.max_new_tokens)
