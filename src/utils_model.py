import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

def initialize_step_tokens(args, model, ref_model, tokenizer, init_token_id=None):
    if not args.use_stp:
        return None

    step_tokens = [f"<step_{i}>" for i in range(args.stp_max_steps)]
    tokenizer.add_tokens(step_tokens, special_tokens=True)

    model.resize_token_embeddings(len(tokenizer))
    if ref_model is not None:
        ref_model.resize_token_embeddings(len(tokenizer))

    step_token_id_map = {
        i: tokenizer.convert_tokens_to_ids(f"<step_{i}>")
        for i in range(args.stp_max_steps)
    }

    new_token_ids = list(step_token_id_map.values())
    old_vocab_size = len(tokenizer) - args.stp_max_steps

    with torch.no_grad():
        model_embeddings = model.get_input_embeddings().weight
        
        if init_token_id is not None:
            init_vector = model_embeddings[init_token_id].clone()
            model_embeddings[new_token_ids] = init_vector
        else:
            mu = model_embeddings[:old_vocab_size].mean()
            sigma = model_embeddings[:old_vocab_size].std()
            model_embeddings[new_token_ids] = torch.empty(
                len(new_token_ids),
                model_embeddings.size(1),
                device=model_embeddings.device,
                dtype=model_embeddings.dtype,
            ).normal_(mean=mu.item(), std=sigma.item())

        if ref_model is not None:
            ref_embeddings = ref_model.get_input_embeddings().weight
            
            if init_token_id is not None:
                ref_init_vector = ref_embeddings[init_token_id].clone()
                ref_embeddings[new_token_ids] = ref_init_vector
            else:
                ref_mu = ref_embeddings[:old_vocab_size].mean()
                ref_sigma = ref_embeddings[:old_vocab_size].std()
                ref_embeddings[new_token_ids] = torch.empty(
                    len(new_token_ids), ref_embeddings.size(1), device=ref_embeddings.device
                ).normal_(mean=ref_mu.item(), std=ref_sigma.item())

    return step_token_id_map

def build_model_tokenizer(args, device):
    print(f"Loading Active Model: {args.model_name}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)
    if ("gpt2" in args.model_name.lower() or
        "neo" in args.model_name.lower() or
        "smol" in args.model_name.lower()):
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if not tokenizer.bos_token_id:
            tokenizer.bos_token_id = tokenizer.eos_token_id
    if not hasattr(tokenizer, 'model_max_length'):
        import pdb; pdb.set_trace()
        tokenizer.model_max_length = ...

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
    ).to(device)

    ref_model = None
    if args.use_kl:
        print(f"Loading Reference Model: {args.model_name}", flush=True)
        ref_model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
        ref_model.eval()

    # This function resizes both model and ref_model
    step_token_id_map = initialize_step_tokens(args,
                                                model,
                                                ref_model,
                                                tokenizer,
                                                init_token_id=tokenizer.eos_token_id if args.init_from_eos else None,
                                                )
                                                
    # Apply the absolute freeze AFTER all matrix resizing
    if args.use_kl and ref_model is not None:
        for p in ref_model.parameters():
            p.requires_grad = False

    if args.use_lora:
        model.gradient_checkpointing_enable()
        peft_config = LoraConfig(
            task_type='CAUSAL_LM',
            target_modules=[
                'q_proj',
                'k_proj',
                'v_proj',
                'o_proj',
                'gate_proj',
                'up_proj',
                'down_proj',
            ],
            modules_to_save=[
                'embed_tokens', 
                'lm_head'
            ],
        )
        model = get_peft_model(model, peft_config)
    
    params_total = sum(p.numel() for p in model.parameters())
    print('params_total', params_total)
    params_learnable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('params_learnable', params_learnable)

    model.train()
    return tokenizer, step_token_id_map, model, ref_model

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


# ---------------------------------------------------------------------------
# Step Token Prediction components  (Section 3.2)
# ---------------------------------------------------------------------------

class StepTokenEmbedding(nn.Module):
    """
    Learned embedding table for M step-token identifiers.

    Each step in the prefix gets one of these embeddings prepended so the
    model can associate content with a discrete identifier.  The completion
    then consists of just these identifiers in the correct topological order.

    The embeddings live in the same space as the transformer's token
    embeddings (dim = d_model) so they can be summed with positional
    encodings by the model's own wpe.
    """

    def __init__(self, n_step_tokens: int, d_model: int):
        super().__init__()
        self.n_step_tokens = n_step_tokens
        self.emb = nn.Embedding(n_step_tokens, d_model)
        # Initialise close to the scale of wte
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def forward(self, indices):
        """
        Args:
            indices: arbitrary shape, values in [0, n_step_tokens).
        Returns:
            Embeddings of the same shape + trailing d_model dim.
        """
        return self.emb(indices)


class StepTokenHead(nn.Module):
    """
    Classification head that maps a hidden state to logits over M step tokens.

    Two modes are supported (controlled at init):
      - 'linear':  single linear projection  D → M
      - 'bilinear':  dot-product retrieval against the StepTokenEmbedding
                      table  (parameter-free beyond the shared embeddings).

    The bilinear mode ties the prediction space to the embedding space,
    which empirically helps when M is small.
    """

    def __init__(self, d_model: int, n_step_tokens: int,
                 mode: str = 'linear', step_token_emb: StepTokenEmbedding = None):
        super().__init__()
        self.mode = mode
        self.n_step_tokens = n_step_tokens

        if mode == 'linear':
            self.proj = nn.Linear(d_model, n_step_tokens)
        elif mode == 'bilinear':
            assert step_token_emb is not None, (
                "bilinear mode requires a reference to the StepTokenEmbedding")
            self.step_token_emb = step_token_emb
            # Optional learned scaling factor
            self.scale = nn.Parameter(torch.tensor(1.0))
        else:
            raise ValueError(f"Unknown StepTokenHead mode: {mode}")

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [..., D]
        Returns:
            logits: [..., M]
        """
        if self.mode == 'linear':
            return self.proj(hidden_states)
        else:
            # Dot product with all M step-token embeddings
            W = self.step_token_emb.emb.weight          # [M, D]
            return self.scale * (hidden_states @ W.T)    # [..., M]


# ---------------------------------------------------------------------------
# Forward helpers
# ---------------------------------------------------------------------------

def forward_no_pos_gpt2(model, input_ids, attention_mask, output_hidden_states=True):
    """
    Forward pass for GPT-2 family with positional embeddings removed.
    Uses token embeddings only: inputs_embeds = wte(input_ids).
    """
    inputs_embeds = model.transformer.wte(input_ids)
    out = model.transformer(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=output_hidden_states,
        use_cache=False,
        return_dict=True,
    )
    return out


def forward_with_step_tokens(model, step_token_emb, batch, output_hidden_states=True):
    """
    Forward pass that injects learned step-token embeddings at marked positions.

    At positions where step_token_mask == 1, the regular wte embedding is
    *replaced* by the corresponding step-token embedding from ``step_token_emb``.
    Positional embeddings (wpe) are still added by the model as usual.

    Args:
        model:          GPT2LMHeadModel (or compatible).
        step_token_emb: StepTokenEmbedding module.
        batch:          dict with at least:
                          input_ids        [B, T]
                          attn_mask        [B, T]
                          step_token_ids   [B, T]   1-indexed step-token id, 0 = regular token
                          step_token_mask  [B, T]   1 at step-token positions
        output_hidden_states: bool

    Returns:
        ModelOutput with .logits, .hidden_states, etc.
    """
    input_ids       = batch['input_ids']
    attn_mask       = batch['attn_mask']
    stp_ids         = batch['step_token_ids']       # 1-indexed; 0 = not a step token
    stp_mask        = batch['step_token_mask']       # binary

    # 1. Regular token embeddings (no positional yet — the model adds wpe)
    token_embeds = model.transformer.wte(input_ids)  # [B, T, D]

    # 2. Step-token embeddings  (convert 1-indexed → 0-indexed; clamp 0→0 is harmless
    #    because those positions are masked out below)
    stp_indices = (stp_ids - 1).clamp(min=0, max=step_token_emb.n_step_tokens - 1)         # [B, T]
    stp_embeds  = step_token_emb(stp_indices)         # [B, T, D]

    # 3. Replace at step-token positions
    mask_f = stp_mask.unsqueeze(-1).float()           # [B, T, 1]
    inputs_embeds = token_embeds * (1.0 - mask_f) + stp_embeds * mask_f

    # 4. Forward through the full model.
    #    Passing inputs_embeds causes GPT2Model to skip wte but still add wpe.
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        output_hidden_states=output_hidden_states,
        use_cache=False,
        return_dict=True,
    )
    return outputs
