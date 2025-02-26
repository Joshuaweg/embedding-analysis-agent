from transformers import AutoTokenizer, AutoModel
import torch

# Load ESM-2 (you can choose different variants)
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

# Example protein sequence
sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

# Tokenize and get embeddings
inputs = tokenizer(sequence, return_tensors="pt")
outputs = model(**inputs)
embeddings = outputs.last_hidden_state
print(embeddings)
