import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import re


class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['input']
        target_text = item['target']

        input_ids = self.tokenizer.encode(
            input_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
        ).squeeze(0)

        target_ids = self.tokenizer.encode(
            target_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
        ).squeeze(0)

        return {"input_ids": input_ids, "target_ids": target_ids}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, ff_hidden_dim):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ff_hidden_dim)
        self.transformer = nn.TransformerEncoder(self.encoder_layers, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding(x)
        x = self.transformer(x)
        return self.fc_out(x)


data = [
    {"input": "What are the symptoms of flu?", "target": "Fever, headache, and cough."},
    {"input": "What is the treatment for flu?", "target": "Rest, hydration, and antiviral medications."},
    {"input": "What are the symptoms of COVID-19?", "target": "Fever, cough, and shortness of breath."},
    {"input": "What is the treatment for COVID-19?", "target": "Supportive care and antiviral medications."}
]

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


dataset = TextDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


model = SimpleTransformer(
    vocab_size=tokenizer.vocab_size,
    d_model=512,
    num_heads=8,
    num_layers=6,
    ff_hidden_dim=2048
)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=5e-5)

num_epochs = 3

for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        input_ids = batch["input_ids"]
        target_ids = batch["target_ids"]


        optimizer.zero_grad()


        outputs = model(input_ids)


        loss = criterion(outputs.view(-1, tokenizer.vocab_size), target_ids.view(-1))
        total_loss += loss.item()


        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")

# -------------------------
torch.save(model.state_dict(), "simple_transformer_model.pth")
print("Model saved as 'simple_transformer_model.pth'")


def generate_text(model, input_text, tokenizer, max_length=50):
    model.eval()
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    with torch.no_grad():
        output = model(input_ids)
    tokens = torch.argmax(output, dim=-1)
    return tokenizer.decode(tokens[0], skip_special_tokens=True)


model.load_state_dict(torch.load("simple_transformer_model.pth"))
print("Model loaded.")

sample_input = "What are the symptoms of flu?"
output_text = generate_text(model, sample_input, tokenizer)
print("Generated Text:", output_text)
