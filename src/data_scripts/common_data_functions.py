
def data_switching(mode, query, p, a, tokenizer):
    """
    Changes data format based on mode input by user
    """
    assert mode in ["q_a", "qp_a", "q_pa","q_p_a"], "select one of the following modes: q_a, qp_a, q_pa, q_p_a"
    sep_token = tokenizer.sep_token
    query,p,a = query.strip(), p.strip(), a.strip()
    if mode == "q_a":
        output =  {'input_ids': query, 'labels': a}
    elif mode == "qp_a":
        query_p = query + " " + sep_token + " " + p
        output =  {'input_ids': query_p, 'labels': a}
    elif mode == "q_pa":
        a_p = p + " " + sep_token + " " + a
        output =  {'input_ids': query, 'decoder_input_ids': a_p}
    elif mode == "q_p_a":
        p = sep_token + " " + p
        output =  {'input_ids': query, 'p_input_ids':p, 'labels': a}
    # Remove samples that are longer than the model max length
    if any([True if len(tokenizer.tokenize(v)) > tokenizer.model_max_length  else False for v in output.values()]): 
        return None
    return output
    
def tokenize_function(inputs, tokenizer):
    """
    For q_pa set the labels to -100 for all tokens that belong to the passage (context)
    """
    tokenized_input = {k: tokenizer(v, truncation = True).input_ids for k,v in inputs.items()}
    if "decoder_input_ids" in tokenized_input:
        labels  = tokenized_input["decoder_input_ids"].copy()
        tokenized_input['labels'] = [[-100 if i < l_i else  l for i,l in enumerate(label)] for label in labels if (l_i:=label.index(tokenizer.sep_token_id)) >0]
    return tokenized_input