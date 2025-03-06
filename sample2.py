import os
import pickle
import torch
import tiktoken
from model import GPTConfig, GPT

# Load the trained model
init_from = 'resume'  # Load the trained model from checkpoint
out_dir = 'out'       # Directory where checkpoint is stored
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

# Load checkpoint
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)

# Load model configuration
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']

# Fix state dict keys if needed
unwanted_prefix = '_orig_mod.'
for k in list(state_dict.keys()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

# Load model state
model.load_state_dict(state_dict)
model.eval()
model.to(device)

# Load tokenizer (GPT-2 encoding)
print("Loading tokenizer...")
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# Chatbot parameters
max_new_tokens = 100
temperature = 0.7
top_k = 50
chat_history = []  # Stores previous messages

print("Chatbot is ready! Type 'exit' to quit.")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    
    # Update chat history
    chat_history.append(f"User: {user_input}")
    
    # Format the chat history as input for the model
    chat_prompt = "\n".join(chat_history) + "\nAI:"
    input_ids = encode(chat_prompt)
    x = torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...]

    # Generate response
    with torch.no_grad():
        output_ids = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    
    # Decode response
    response = decode(output_ids[0].tolist()).strip()
    
    # Extract AI response (avoid repeating input)
    response = response[len(chat_prompt):].strip()
    
    # Print AI response
    print(f"AI: {response}")
    
    # Append response to chat history
    chat_history.append(f"AI: {response}")

    # Keep context manageable
    if len(chat_history) > 10:  # Keep last 10 exchanges
        chat_history = chat_history[-10:]
