import sys
import pickle
import numpy as np
import re
from model import DeepTernaryNetworkMHot

def main(prompt, generate_length=10):
    with open("dataset.pkl", "rb") as f:
        data = pickle.load(f)
        
    word_to_id = data["word_to_id"]
    id_to_word = data["id_to_word"]
    vocab_size = len(word_to_id)
    ctx_size = data["context_size"]

    model = DeepTernaryNetworkMHot(vocab_size, [64, 64], vocab_size)
    model.load_weights("weights.pkl")

    # Clean and tokenize prompt
    prompt_words = re.sub(r'[^\w\s]', '', prompt.lower()).split()
    
    if len(prompt_words) < ctx_size:
        print(f"Error: Prompt must be at least {ctx_size} words long.")
        return

    # Seed the generation context
    context = []
    for w in prompt_words[-ctx_size:]:
        if w in word_to_id:
            context.append(word_to_id[w])
        else:
            # Fallback for unknown words
            context.append(np.random.choice(list(word_to_id.values()))) 

    generated_text = list(prompt_words)

    # Autoregressive Generation
    for _ in range(generate_length):
        y_pred = model.forward(context)
        
        # Greedily pick the highest probability word
        next_word_id = np.argmax(y_pred)
        next_word = id_to_word[next_word_id]
        
        generated_text.append(next_word)
        
        # Slide the context window
        context.append(next_word_id)
        context = context[1:]

    print("\n--- Generated Text ---")
    print(" ".join(generated_text))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python infer.py "your starting prompt here"')
    else:
        main(sys.argv[1])