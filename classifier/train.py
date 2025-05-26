
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import os


torch.manual_seed(42)
NUM_EPOCHS = 5
BATCH_SIZE = 16
FOUNDATION_MODEL = "Qwen/Qwen2.5-0.5B"
DATASET = "ucberkeley-dlab/measuring-hate-speech"
MODELS_DIR = 'classifier/models'

def iterate_over_dataloader(model, tokenizer, optimizer, dataloader, split):
    
    total_loss = 0

    for batch in tqdm(dataloader):
        tokenized_input = tokenizer(batch['text'], padding=True, return_tensors='pt')
        output = model(tokenized_input['input_ids'], tokenized_input['attention_mask'])
        output_logits = output.logits[:, 0]  # indexing reduces to 1d

        hate_speech_mask = batch['hate_speech_score'] >= 0.5
        race_mask = batch['target_race'] == 1
        racist_speech_targets = hate_speech_mask & race_mask

        loss = F.binary_cross_entropy_with_logits(output_logits, racist_speech_targets.float())
        total_loss += loss.item()

        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def main():
    if not os.path.isdir(MODELS_DIR):
        print(f'{MODELS_DIR} directory does not exist. Canceling training run.')
        return

    train_dataset = load_dataset(DATASET, split="train[:70%]")
    val_dataset = load_dataset(DATASET, split="train[70%:85%]")
    tokenizer = AutoTokenizer.from_pretrained(FOUNDATION_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        FOUNDATION_MODEL,
        num_labels=1,
        sliding_window=None,
        pad_token_id=tokenizer.pad_token_id
    )
    optimizer = AdamW(model.parameters())

    for i in range(NUM_EPOCHS):
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        train_loss = iterate_over_dataloader(model, tokenizer, optimizer, train_dataloader, 'train')
        print(f'Epoch {i}, Train loss: {train_loss}')
        with torch.no_grad():
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loss = iterate_over_dataloader(model, tokenizer, optimizer, val_dataloader, 'val')
        print(f'Epoch {i}, Val loss: {val_loss}')

    torch.save(
        model.state_dict(),
        f'classifier/models/trained_classifier.pt'
    )

if __name__ == '__main__':
    main()
