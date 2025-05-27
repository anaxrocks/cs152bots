import torch.nn as nn

class Message:
    self.file_path
    self.is_audio
    self.content

class RacistClassifier(nn.Module):
    def __init__():
        self.model = AutoModelForSequenceClassification.from_pretrained(
            FOUNDATION_MODEL,
            num_labels=1,
            sliding_window=None,
            pad_token_id=tokenizer.pad_token_id
        ).to(dtype=torch.float16, device=device)
        # load state_dict
        # load state_dict into model

    def make_prediction(messages):
        text = ""
        for message in messages:
            if message.is_audio:
                transcription = transcript_model(message.file_path)
                text += transcription
            else:
                text += message.content
        
        tokenized_text = tokenizer(text)
        pred_logit = self.model(tokenized_text)
        pred_probability = softmax(pred_logit)
        return pred_probability > 0.5
