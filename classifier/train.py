from datasets import load_dataset, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import os
import argparse
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import sklearn.metrics as metrics

FOUNDATION_MODEL = "Qwen/Qwen2.5-0.5B"
DATASET = "ucberkeley-dlab/measuring-hate-speech"
DATASET_SEED = 42
MODELS_DIR = 'classifier/models'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calc_metrics(logits, targets, threshold):
    probs = torch.sigmoid(torch.Tensor(logits))
    preds = probs >= threshold
    acc = metrics.accuracy_score(targets, preds)
    print(f'Accuracy: {acc}')
    precision = metrics.precision_score(targets, preds)
    print(f'Precision: {precision}')
    recall = metrics.recall_score(targets, preds)
    print(f'Recall: {recall}')
    f1_score = metrics.f1_score(targets, preds)
    print(f'F1 Score: {f1_score}')

def balanced_class_weights():
    dataset = load_dataset(DATASET, split='train')  # train split includes all rows
    hate_speech_mask = np.array(dataset['hate_speech_score']) >= 0.5
    race_mask = np.array(dataset['target_race']) == 1
    racist_speech_targets = hate_speech_mask & race_mask
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(racist_speech_targets), y=racist_speech_targets)
    not_racist_weight, racist_weight = weights
    return not_racist_weight, racist_weight

def iterate_over_dataloader(model, tokenizer, optimizer, dataloader, training, not_racist_weight, racist_weight, threshold):
    total_loss = 0
    all_logits = []
    all_targets = []

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

        all_logits.extend(output_logits.tolist())
        all_targets.extend(racist_speech_targets.tolist())

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    calc_metrics(all_logits, all_targets, threshold)
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def main(args):
    if not os.path.isdir(MODELS_DIR):
        print(f'{MODELS_DIR} directory does not exist. Canceling training run.')
        return

    full_dataset = load_dataset(DATASET, split="train", columns=["text", "hate_speech_score", "target_race"]).shuffle(DATASET_SEED)
    train_end_idx = int(len(full_dataset) * 0.70)
    val_end_idx = int(len(full_dataset) * 0.85)
    train_dataset = full_dataset[:train_end_idx]
    val_dataset = full_dataset[train_end_idx:val_end_idx]
    tokenizer = AutoTokenizer.from_pretrained(FOUNDATION_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        FOUNDATION_MODEL,
        num_labels=1,
        sliding_window=None,
        pad_token_id=tokenizer.pad_token_id
    ).to(device=DEVICE)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    not_racist_weight, racist_weight = balanced_class_weights()

    best_val = float('inf')
    for i in range(args.epochs):
        train_dataloader = torch.utils.data.DataLoader(Dataset.from_dict(train_dataset), batch_size=args.batch_size, shuffle=True)
        train_loss = iterate_over_dataloader(
            model,
            tokenizer,
            optimizer,
            train_dataloader,
            True,
            not_racist_weight,
            racist_weight,
            args.threshold
        )
        print(f'Epoch {i}, Train loss: {train_loss}')
        with torch.no_grad():
            val_dataloader = torch.utils.data.DataLoader(Dataset.from_dict(val_dataset), batch_size=args.batch_size, shuffle=True)
            val_loss = iterate_over_dataloader(
                model,
                tokenizer,
                optimizer,
                val_dataloader,
                False,
                not_racist_weight,
                racist_weight,
                args.threshold
            )
        print(f'Epoch {i}, Val loss: {val_loss}')
        if val_loss < best_val:
            print('Saving model.')
            torch.save(
                model.state_dict(),
                f'{MODELS_DIR}/seed-{args.seed}_epochs-{args.epochs}_batch-{args.batch_size}_lr-{args.learning_rate}.pt'
            )
            best_val = val_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--threshold', default=0.5, type=float)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    main(args)
