from datasets import load_dataset
from train import DATASET, DATASET_SEED
import numpy as np

def calc_percentages(dataset, split):
    print(split)
    num_total = len(dataset['hate_speech_score'])
    print(f'Total: {num_total}')

    hate_speech_mask = np.array(dataset['hate_speech_score']) >= 0.5
    race_mask = np.array(dataset['target_race']) == 1
    racist_speech_targets = hate_speech_mask & race_mask
    num_target = racist_speech_targets.sum()
    print(f'Target: {num_target}')

    print(f'Percentage of total: {num_target / num_total}\n')

full_dataset = load_dataset(DATASET, split="train", columns=["text", "hate_speech_score", "target_race"]).shuffle(DATASET_SEED)
train_end_idx = int(len(full_dataset) * 0.70)
val_end_idx = int(len(full_dataset) * 0.85)
train_dataset = full_dataset[:train_end_idx]
val_dataset = full_dataset[train_end_idx:val_end_idx]
test_dataset = full_dataset[val_end_idx:]
calc_percentages(train_dataset, 'Train')
calc_percentages(val_dataset, 'Val')
calc_percentages(test_dataset, 'Test')
