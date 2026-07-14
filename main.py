import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import csv
import os
from transformer_architecture import Transformer

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    CSV_FILE = 'data.csv'
    
    csv_data = []
    with open(CSV_FILE, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            csv_data.append(row)

    class Vocabulary:
        def __init__(self):
            self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
            self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        
        def __len__(self):
            return len(self.itos)

        # Here it loops through every word in every sentence 
        # and adds an unique index starting from 4 to every other words
        # other than those 4 tokens: <PAD>, <SOS>, <EOS>, <UNK>
        def build_vocab(self, sentences):
            idx = 4
            for sentence in sentences:
                for word in sentence.lower().split():
                    if word not in self.stoi:
                        self.stoi[word] = idx
                        self.itos[idx] = word
                        idx += 1
        
        # Here it converts the text into a list of indices and if 
        # the word is not in the vocabulary it returns <UNK>
        def numericalize(self, text):
            return [self.stoi.get(token, self.stoi["<UNK>"]) for token in text.lower().split()]

    # Read from 2nd line because 1st line can be a header
    eng_sentences = [row[0] for row in csv_data[1:]]
    ben_sentences = [row[1] for row in csv_data[1:]]

    # Build the vocabulary
    src_vocab = Vocabulary()
    src_vocab.build_vocab(eng_sentences)
    tgt_vocab = Vocabulary()
    tgt_vocab.build_vocab(ben_sentences)

    # Prepare src & tgt indices
    src_indices = []
    tgt_indices = []

    # Here it adds <SOS> at the start and <EOS> at the end of every sentence
    for src_text, tgt_text in zip(eng_sentences, ben_sentences):
        s_idx = [src_vocab.stoi["<SOS>"]] + src_vocab.numericalize(src_text) + [src_vocab.stoi["<EOS>"]]
        t_idx = [tgt_vocab.stoi["<SOS>"]] + tgt_vocab.numericalize(tgt_text) + [tgt_vocab.stoi["<EOS>"]]
        
        src_indices.append(torch.tensor(s_idx))
        tgt_indices.append(torch.tensor(t_idx))

    # Pad all sentences to match the longest one by adding <PAD>
    src_batch = pad_sequence(src_indices, padding_value=src_vocab.stoi["<PAD>"], batch_first=True).to(device)
    tgt_batch = pad_sequence(tgt_indices, padding_value=tgt_vocab.stoi["<PAD>"], batch_first=True).to(device)

    print(f"Data Loaded. Batch Shape: {src_batch.shape}")

    # Transformer setup
    model = Transformer(src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab), src_pad_idx=src_vocab.stoi["<PAD>"], tgt_pad_idx=tgt_vocab.stoi["<PAD>"],
    embed_size=64, num_layers=2, forward_expansion=2, heads=4, dropout=0.1, device=device, max_length=100).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.stoi["<PAD>"])

    # Training 
    print("\nStarting Training...")
    model.train()

    for epoch in range(500):
        optimizer.zero_grad()
        
        # Implementation of 'Teacher Forcing' in a Seq2Seq transformer
        output = model(src_batch, tgt_batch[:, :-1])
        
        # Reshape
        output = output.reshape(-1, len(tgt_vocab))
        target = tgt_batch[:, 1:].reshape(-1)
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    # Testing
    print("Testing->")
    print()

    model.eval()

    test_sentences = [
        "We study machine learning",
        "Arjun bought a new computer",
        "She drinks tea",
        "she read machine learning",
        "We play ludo",
        "Rohan plays football",
        "Puja eats rice",
        "We drink tea"
    ]

    def translate_sentence(sentence, model, src_vocab, tgt_vocab, device, max_len=20):
        # Tokenize and add <SOS>/<EOS>
        src_idx = [src_vocab.stoi["<SOS>"]] + src_vocab.numericalize(sentence) + [src_vocab.stoi["<EOS>"]]
        src_tensor = torch.LongTensor(src_idx).unsqueeze(0).to(device) # Shape: (1, seq_len)

        # Start with <SOS>
        outputs = [tgt_vocab.stoi["<SOS>"]]
        
        for _ in range(max_len):
            trg_tensor = torch.LongTensor([outputs]).to(device)

            with torch.no_grad():
                output = model(src_tensor, trg_tensor)

            # Get the token with highest probability from the last step
            best_guess = output[:, -1, :].argmax(1).item()

            # Stop if model predicts EOS
            if best_guess == tgt_vocab.stoi["<EOS>"]:
                break

            outputs.append(best_guess)
        
        # Convert indices back to words (skipping <SOS>)
        return " ".join([tgt_vocab.itos[idx] for idx in outputs[1:]])

    for i, sentence in enumerate(test_sentences):
        translation = translate_sentence(sentence, model, src_vocab, tgt_vocab, device)
        print(f"Input {i+1}: {sentence}")
        print(f"Output {i+1}: {translation}")
        print()

