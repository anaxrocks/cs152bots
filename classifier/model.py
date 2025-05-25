import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

BASE_MODEL = "Qwen/Qwen2.5-0.5B"

# FULL DISCLOSURE: I (Matt Wolff) am also finetuning Qwen as part of my CS224R final project.
# In that project, I am finetuning Qwen to train a Bradley-Terry reward model, which is
# quite different from training a hate speech classifier like we do in this project.
# However, the following code defining the model to be finetuned below is exactly the same as from
# my other project.

class HateSpeechClassifier(nn.Module):
    def __init__(self):
        super(HateSpeechClassifier, self).__init__()

        self.base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, sliding_window=None)
        hidden_dim = self.base_model.config.hidden_size
        self.output_layer = nn.Linear(hidden_dim, 1)  # todo: initialize layer

        parameters = list(self.base_model.parameters()) + list(self.output_layer.parameters())
        self.optimizer = torch.optim.AdamW(parameters)

    def forward(self, input_ids, attn_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        return self.output_layer(last_hidden_state)

hate_speech_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)