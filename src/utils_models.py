def prep_inputs_for_causal_lm(input_ids, attention_mask):
    '''
    Standard HF approach:
    - Input and Labels are the same length.
    - Padding in labels is masked with -100.
    - Shifting is handled in the training loop or model.
    '''
    labels = input_ids.clone()
    # Mask padding tokens
    labels[attention_mask == 0] = -100
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }