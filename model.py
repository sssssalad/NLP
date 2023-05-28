import torch.nn as nn
from transformers import AutoModel
from src.fine_tune.torchcrf import CRF

class BNER(nn.Module):
    def __init__(self, BioBERT_path, model_config):
        super(BNER, self).__init__()
        self.BioBERT = AutoModel.from_pretrained(BioBERT_path)
        self.dropout = nn.Dropout(model_config.dropout)
        self.classifier = nn.Linear(in_features=model_config.hidden_size,
                                    out_features=model_config.num_label)
        self.crf = CRF(model_config.num_label, batch_first=True)


    def forward(self, token_id, attention_mask, segment_ids, label_id):
        emissions = self.get_emissions(token_id, attention_mask, segment_ids)
        loss = -1 * self.crf(emissions, label_id, attention_mask.byte())
        return loss

    def predict(self,token_id, attention_mask, segment_ids):
        emissions = self.get_emissions(token_id, attention_mask, segment_ids)
        # best_tag = self.crf.decode(emissions, attention_mask.byte())

        return emissions

    def get_emissions(self, token_id, attention_mask, segment_ids):
        BioBERT_embeds = self.BioBERT(token_id, attention_mask, segment_ids)[0]
        out = self.dropout(BioBERT_embeds)
        emissions = self.classifier(out)
        return emissions
