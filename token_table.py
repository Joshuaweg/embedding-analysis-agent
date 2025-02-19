from transformers import GPT2Tokenizer
import json

def create_token_table():
    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Get the vocabulary with IDs
    vocab = tokenizer.get_vocab()
    
    # Create a sorted list of (id, token) pairs
    token_table = [(id, token) for token, id in vocab.items()]
    token_table.sort()  # Sort by ID
    
    # Save as JSON for easy lookup
    with open('token_table.json', 'w', encoding='utf-8') as f:
        json.dump({str(id): token for id, token in token_table}, f, ensure_ascii=False, indent=2)
    
    # Save as readable text file
    with open('token_table.txt', 'w', encoding='utf-8') as f:
        f.write("ID\tToken\tPrintable Representation\n")
        f.write("-" * 50 + "\n")
        for id, token in token_table:
            # Create a printable representation for special characters
            printable = repr(token)[1:-1]  # Remove quotes
            f.write(f"{id}\t{token}\t{printable}\n")

def lookup_tokens(id_string):
    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Convert space-separated string to list of integers
    try:
        ids = [int(id) for id in id_string.split()]
    except ValueError:
        print("Error: Please enter valid numbers separated by spaces")
        return
    
    # Get tokens for each ID
    print("\nID\tToken\tBytes\tDetails")
    print("-" * 60)
    for id in ids:
        token = tokenizer.decode([id])
        # Get the raw token from vocab (to see special characters)
        raw_token = next(k for k, v in tokenizer.get_vocab().items() if v == id)
        # Get byte representation
        bytes_repr = ' '.join(f'{byte:02x}' for byte in raw_token.encode('utf-8'))
        
        details = []
        if raw_token.startswith('Ġ'):
            details.append("has leading space")
        if raw_token != token:
            details.append(f"raw form: {repr(raw_token)[1:-1]}")
            
        print(f"{id}\t{token}\t{bytes_repr}\t{', '.join(details)}")

if __name__ == "__main__":
    while True:
        print("\nEnter token IDs (space-separated) or 'q' to quit:")
        user_input = input().strip()
        
        if user_input.lower() == 'q':
            break
            
        lookup_tokens(user_input) 