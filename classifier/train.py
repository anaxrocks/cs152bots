
from datasets import load_dataset
from model import HateSpeechClassifier, hate_speech_tokenizer
import torch


torch.manual_seed(42)
NUM_EPOCHS = 5
BATCH_SIZE = 16
DATASET = "ucberkeley-dlab/measuring-hate-speech"

train_dataset = load_dataset(DATASET, split="train[:70%]")
val_dataset = load_dataset(DATASET, split="train[70%:85%]")
model = HateSpeechClassifier()

for i in range(NUM_EPOCHS):
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    for batch in dataloader:
        tokenized_input = hate_speech_tokenizer(batch['text'], padding=True, return_tensors='pt')
        output = model(tokenized_input['input_ids'], tokenized_input['attention_mask'])
        
        hate_speech_mask = batch['hate_speech_score'] >= 0.5
        race_mask = batch['target_race'] >= 0.5
        racist_speech_targets = hate_speech_mask & race_mask

        

