import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
from common_data_functions import data_switching, tokenize_function
from datasets import Dataset


def process_dataset(data_path, tokenizer, mode):
    data = pd.read_csv(data_path).dropna()
    process_dataset = []
    for query, passage, answer in tqdm(zip(data["instruction"].values, data["context"].values, data["response"].values)):
            inputs = data_switching(mode, query, passage, answer, tokenizer)
            if inputs is not None:
                process_dataset.append(inputs)
            else:
                 continue
    df = Dataset.from_list(process_dataset)
    df = df.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer}, remove_columns= df.column_names)
    return df


if __name__ == "__main__":
    data_path = r"C:\Users\Daniel\Documents\Effiecient_Knowledge_Grounding\data\gpt_generated\gpt_dolly.csv"
    model_name = "t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'sep_token': '<sep>'})
    for mode in ["q_a", "qp_a", "q_pa","q_p_a"]:
        df = process_dataset(data_path, tokenizer, mode=mode)
        df = df.train_test_split(test_size=1_000, shuffle=True)
        df.save_to_disk(rf"C:\Users\Daniel\Documents\Effiecient_Knowledge_Grounding\data\gpt_generated\train_data_{model_name.replace('/','_')}_{mode}")
        print(df)