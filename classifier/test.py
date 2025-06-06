import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from train import FOUNDATION_MODEL, DATASET, DATASET_SEED, DEVICE, MODELS_DIR, iterate_over_dataloader, balanced_class_weights
import torch
from datasets import load_dataset, Dataset

def main(args):
    model_path = f'{MODELS_DIR}/seed-{args.seed}_epochs-{args.epochs}_batch-{args.batch_size}_lr-{args.learning_rate}.pt'
    state_dict = torch.load(model_path, map_location=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(FOUNDATION_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        FOUNDATION_MODEL,
        num_labels=1,
        sliding_window=None,
        pad_token_id=tokenizer.pad_token_id
    ).to(DEVICE)
    print('Loading state dict.')
    model.load_state_dict(state_dict)
    not_racist_weight, racist_weight = balanced_class_weights()

    full_dataset = load_dataset(DATASET, split="train", columns=["text", "hate_speech_score", "target_race"]).shuffle(DATASET_SEED)
    val_end_idx = int(len(full_dataset) * 0.85)
    test_dataset = full_dataset[val_end_idx:]
    with torch.no_grad():
        test_dataloader = torch.utils.data.DataLoader(Dataset.from_dict(test_dataset), batch_size=args.batch_size, shuffle=True)
        test_loss = iterate_over_dataloader(
            model,
            tokenizer,
            None,
            test_dataloader,
            False,
            not_racist_weight,
            racist_weight,
            args.threshold
        )
    print(f'Test loss: {test_loss}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--threshold', default=0.5, type=float)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    main(args)
