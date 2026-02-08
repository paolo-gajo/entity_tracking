import torch

def prep_inputs_for_causal_lm(labels, attention_mask, bos_token_id):
    '''
    Prepares right-shifted inputs for causal language modeling from labels.
    '''
    input_ids = torch.zeros_like(labels)
    input_ids[:, 1:] = labels[:, :-1]
    input_ids[:, 0] = bos_token_id
    labels[attention_mask == 0] = -100
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }