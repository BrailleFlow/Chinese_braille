import os
from transformers import MT5Tokenizer, MT5ForConditionalGeneration

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "down_model")
OUTPUT_DIR = MODEL_DIR  # same directory

# Check directory
def directory_exists_and_not_empty(path):
    return os.path.exists(path) and os.path.isdir(path) and len(os.listdir(path)) > 0

if not directory_exists_and_not_empty(MODEL_DIR):
    print(f"Directory {MODEL_DIR} does not exist or is empty. Downloading mT5-small...")
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model and tokenizer saved to {OUTPUT_DIR}")
else:
    print(f"Model directory {MODEL_DIR} exists and is not empty. No action needed.")

# Add Braille characters as tokens
model = MT5ForConditionalGeneration.from_pretrained(MODEL_DIR)
tokenizer = MT5Tokenizer.from_pretrained(MODEL_DIR)

braille_chars = "⠂⠆⠒⠲⠢⠖⠶⠦⠔⠴⠁⠃⠉⠙⠑⠋⠛⠓⠊⠚⠅⠇⠍⠝⠕⠏⠟⠗⠎⠞⠥⠧⠺⠭⠽⠵⠮⠐⠼⠫⠩⠯⠄⠷⠾⠡⠬⠠⠤⠨⠌⠆⠰⠣⠿⠜⠹⠈⠪⠳⠻⠘⠸"
new_tokens = list(braille_chars)
num_added_toks = tokenizer.add_tokens(new_tokens)
print(f"Number of tokens added: {num_added_toks}")

model.resize_token_embeddings(len(tokenizer))

tokenizer.save_pretrained(OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)
print(f"Model and tokenizer updated and saved to {OUTPUT_DIR}")
