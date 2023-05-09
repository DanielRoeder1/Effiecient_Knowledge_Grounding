import json
from datasets import Dataset
from tqdm import tqdm

def get_dataset(data_path, tokenizer, mode="q_a"):
    assert mode in ["q_a", "qp_a", "q_pa","q_p_a"], "select one of the following modes: q_a, qp_a, q_pa"
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
                    query,p,a = query.strip(), p.strip(), a.strip()
                    if mode == "q_a":
                        dataset.append({'input_ids': query, 'labels': a})
                    elif mode == "qp_a":
                        query_p = query + " " + sep_token + " " + p
                        dataset.append({'input_ids': query_p, 'labels': a})
                    elif mode == "q_pa":
                        a_p = p + " " + sep_token + " " + a
                        dataset.append({'input_ids': query, 'decoder_input_ids': a_p})
                    elif mode == "q_p_a":
                        dataset.append({'input_ids': query, 'p_input_ids':p, 'labels': a})
    return Dataset.from_list(dataset)

def tokenize_function(inputs, tokenizer):
    """
    For q_pa set the labels to -100 for all tokens that belong to the passage (context)
    """
    tokenized_input = {k: tokenizer(v, truncation = True).input_ids for k,v in inputs.items()}
    if "decoder_input_ids" in tokenized_input:
        labels  = tokenized_input["decoder_input_ids"].copy()
        tokenized_input['labels'] = [[-100 if i < l_i else  l for i,l in enumerate(label)] for label in labels if (l_i:=label.index(tokenizer.sep_token_id)) >0]
    return tokenized_input

    
if __name__ == "__main__":
    from transformers import AutoTokenizer
    from datasets import concatenate_datasets
    model_name = "t5-base"
    model_name = "google/t5-v1_1-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'sep_token': '<sep>'})
    
    for mode in ["q_a", "qp_a", "q_pa","q_p_a"]:
        train_data = get_dataset(r'C:\Users\Daniel\Documents\Effiecient_Knowledge_Grounding\data\MSMacro\train_v2.1.json', tokenizer, mode=mode)
        dev_data = get_dataset(r'C:\Users\Daniel\Documents\Effiecient_Knowledge_Grounding\data\MSMacro\dev_v2.1.json', tokenizer, mode=mode)
        # Because we potentially have multiple entries for each q,a pair (one for each passage) we split the dev set seperately to minmize leakage
        #data_all = dev_data
        #data_all["train"] = concatenate_datasets([train_data, data_all["train"]])
        data_all = concatenate_datasets([train_data, dev_data])
        data_all = data_all.train_test_split(test_size= 10_000, train_size= 30_000, shuffle=False)

        data_all.save_to_disk(rf'C:\Users\Daniel\Documents\Effiecient_Knowledge_Grounding\data\MSMacro\train_test__{model_name.replace("/","_")}_{mode}')
        print(data_all)

        