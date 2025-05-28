from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import os
import argparse
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

FOUNDATION_MODEL = "Qwen/Qwen2.5-0.5B"
DATASET = "ucberkeley-dlab/measuring-hate-speech"
MODELS_DIR = 'classifier/models'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def balanced_class_weights():
    dataset = load_dataset(DATASET, split='train')  # train split includes all rows
    hate_speech_mask = np.array(dataset['hate_speech_score']) >= 0.5
    race_mask = np.array(dataset['target_race']) == 1
    racist_speech_targets = hate_speech_mask & race_mask
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(racist_speech_targets), y=racist_speech_targets)
    not_racist_weight, racist_weight = weights
    return not_racist_weight, racist_weight

def iterate_over_dataloader(model, tokenizer, optimizer, dataloader, training, not_racist_weight, racist_weight):
    total_loss = 0

    for batch in tqdm(dataloader):
        tokenized_input = tokenizer(batch['text'], padding=True, return_tensors='pt')
        output = model(tokenized_input['input_ids'].to(device=DEVICE), tokenized_input['attention_mask'].to(device=DEVICE))
        output_logits = output.logits[:, 0]  # indexing reduces to 1d

        hate_speech_mask = batch['hate_speech_score'] >= 0.5
        race_mask = batch['target_race'] == 1
        racist_speech_targets = (hate_speech_mask & race_mask).to(device=DEVICE)

        loss = F.binary_cross_entropy_with_logits(output_logits, racist_speech_targets.float(), reduction='none')
        loss[racist_speech_targets.logical_not()] *= not_racist_weight  # weights account for imbalanced dataset
        loss[racist_speech_targets] *= racist_weight
        loss = loss.mean()
        total_loss += loss.item()

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def main(args):
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
    ).to(device=DEVICE)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    not_racist_weight, racist_weight = balanced_class_weights()

    for i in range(args.epochs):
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        train_loss = iterate_over_dataloader(
            model,
            tokenizer,
            optimizer,
            train_dataloader,
            True,
            not_racist_weight,
            racist_weight,
        )
        print(f'Epoch {i}, Train loss: {train_loss}')
        with torch.no_grad():
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
            val_loss = iterate_over_dataloader(
                model,
                tokenizer,
                optimizer,
                val_dataloader,
                False,
                not_racist_weight,
                racist_weight,
            )
        print(f'Epoch {i}, Val loss: {val_loss}')

    torch.save(
        model.state_dict(),
        f'{MODELS_DIR}/seed-{args.seed}_epochs-{args.epochs}_batch-{args.batch_size}_lr-{args.learning_rate}.pt'
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    args = parser.parse_args()
    torch.manual_seed(parser.seed)
    main(args)
