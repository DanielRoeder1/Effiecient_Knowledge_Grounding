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
    tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-base")
    tokenizer.add_special_tokens({'sep_token': '<sep>'})
    mode = "q_pa"
    df = process_dataset(data_path, tokenizer, mode=mode)
    df.save_to_disk(rf"C:\Users\Daniel\Documents\Effiecient_Knowledge_Grounding\data\gpt_generated\train_data_{len(df)}_{mode}")
    print(df)