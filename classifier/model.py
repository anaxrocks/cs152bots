import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import whisper
from typing import List
import warnings

warnings.filterwarnings("ignore")
FOUNDATION_MODEL = "Qwen/Qwen2.5-0.5B"
MODEL_PATH = "../classifier/models/seed-42_epochs-3_batch-24_lr-1e-05.pt"
DEVICE = torch.device("cpu")
transcript_model = whisper.load_model("base")
tokenizer = AutoTokenizer.from_pretrained(FOUNDATION_MODEL)

class ModelMessage:
    def __init__(self, *, file_path: str = None, content: str = None):
        self.file_path = file_path
        self.content = content
        self.is_audio = file_path is not None

class RacistClassifier(nn.Module):
    def __init__(self, model_path: str = MODEL_PATH):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            FOUNDATION_MODEL,
            num_labels=1,
            pad_token_id=tokenizer.pad_token_id
        ).to(device=DEVICE)
        
        try:
            state_dict = torch.load(model_path, map_location=DEVICE)
            self.model.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")

        self.eval()

    def make_prediction(self, messages: List[ModelMessage]) -> bool:
        text = ""
        for message in messages:
            if message.is_audio:
                try:
                    result = transcript_model.transcribe(message.file_path)
                    transcription = result["text"]
                    print(f"[TRANSCRIPT LOG] Audio message content:\n{transcription}\n")
                    text += transcription + " "
                except Exception as e:
                    print(f"[Whisper Error] {message.file_path}: {e}")
            else:
                text += message.content + " "

        if not text.strip():
            return False  # Skip empty input

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, 0]
            probability = torch.sigmoid(logits).item()
            print(f"[CLASSIFICATION LOG] Probability of hate speech: {probability:.4f}")
            return probability > 0.5

    def mock_transcribe_audio(self, file_path: str) -> str:
        try:
            result = transcript_model.transcribe(file_path)
            return result["text"]
        except Exception as e:
            print(f"[Whisper Error - mock]: {e}")
            return "[Transcription failed]"
