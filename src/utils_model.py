# srfc/utils_model.py
import torch.nn as nn

class PositionHead(nn.Module):
    def __init__(self, d_model: int, n_bins: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_bins),
        )

    def forward(self, x):
        return self.net(x)

def forward_no_pos_gpt2(model, input_ids, attention_mask, output_hidden_states=True):
    """
    Forward pass for GPT-2 family with positional embeddings removed.
    Uses token embeddings only: inputs_embeds = wte(input_ids).
    """
    inputs_embeds = model.transformer.wte(input_ids)  # [B, T, D]
    out = model.transformer(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=output_hidden_states,
        use_cache=False,
        return_dict=True,
    )
    # out.last_hidden_state: [B, T, D]
    # out.hidden_states: tuple(len = n_layers+1) if requested
    return out