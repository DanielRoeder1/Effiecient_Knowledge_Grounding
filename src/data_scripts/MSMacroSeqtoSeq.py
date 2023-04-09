import json
from datasets import Dataset
from tqdm import tqdm

def get_dataset(data_path, tokenizer, mode="q_a"):
    assert mode in ["q_a", "qp_a", "q_pa"], "select one of the following modes: q_a, qp_a, q_pa"
    if mode in ["q_pa", "qp_a"]:
        assert tokenizer.sep_token_id is not None, "tokenizer must have a sep_token_id for mode q_pa or qp_a"
    with open(data_path) as f:
        data = json.load(f)
    data = transform(data, mode, tokenizer.sep_token)
    data = data.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer}, remove_columns= data.column_names)
    return data

def transform(data, mode, sep_token):
    """
    Only keep entries that contain a well formed answer
    Create one entry for each combination of passage and well formed answer with the same query
    """
    dataset = []
    for query, passage, answer in tqdm(zip(data["query"].values(), data["passages"].values(), data["wellFormedAnswers"].values())):
        if answer != "[]" and len(select_p:= [p['passage_text'] for p in passage if p['is_selected'] == 1]) > 0:
            if mode == "q_a": select_p = [select_p[0]]
            for p in select_p:
                for a in answer:
                    if mode == "q_a":
                        dataset.append({'input_ids': query, 'labels': a})
                    elif mode == "qp_a":
                        query = query + sep_token + p
                        dataset.append({'input_ids': query, 'labels': a})
                    elif mode == "q_pa":
                        a = p + sep_token + a
                        dataset.append({'input_ids': query, 'decoder_input_ids': a})
                    elif mode == "q_p_a":
                        dataset.append({'input_ids': query, 'p_input_ids':p, 'labels': a})
    return Dataset.from_list(dataset)

def tokenize_function(inputs, tokenizer):
    """
    For q_pa set the labels to -100 for all tokens that belong to the passage (context)
    """
    tokenized_input = {k: tokenizer(v).input_ids for k,v in inputs.items()}
    if "decoder_input_ids" in tokenized_input:
        labels  = tokenized_input["decoder_input_ids"].copy()
        tokenized_input['labels'] = [[-100 if i < l_i else  l for i,l in enumerate(label)] for label in labels if (l_i:=label.index(tokenizer.sep_token_id)) >0]
    return tokenized_input

    
if __name__ == "__main__":
    from transformers import AutoTokenizer
    from datasets import concatenate_datasets

    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    tokenizer.add_special_tokens({'sep_token': '<sep>'})
    
    train_data = get_dataset(r'C:\Users\Daniel\Documents\Effiecient_Knowledge_Grounding\data\train_v2.1.json', tokenizer, mode="q_pa")
    dev_data = get_dataset(r'C:\Users\Daniel\Documents\Effiecient_Knowledge_Grounding\data\dev_v2.1.json', tokenizer, mode="q_pa")
    data_all = concatenate_datasets([train_data, dev_data])
    data_all = data_all.train_test_split(test_size= 10_000, shuffle=True, seed=42)

    data_all.save_to_disk(r'C:\Users\Daniel\Documents\Effiecient_Knowledge_Grounding\data\train_test_q_pa')

        