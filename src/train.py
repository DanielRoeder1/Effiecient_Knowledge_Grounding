from utils.utils import load_config
from utils.training_utils import get_optimizer
from data_scripts.Collator import DataCollatorForPromptedSeq2Seq

import os
from datasets import load_from_disk
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer, T5ForConditionalGeneration


def train():
    dir_path = os.path.dirname(__file__)    
    config_path = os.path.join(dir_path, "config/config.yaml")
    args = load_config(config_path)

    dataset = load_from_disk(args.paths.data_path)
    if args.optim.total_steps is None: args.optim.total_steps = len(dataset["train"]) * args.train.batch_size * args.train.epochs
    model = T5ForConditionalGeneration.from_pretrained(args.model.path).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model.path)
    tokenizer.add_special_tokens({'sep_token': args.model.sep_token})
    model.resize_token_embeddings(len(tokenizer))
    collate_fn = DataCollatorForPromptedSeq2Seq(tokenizer, model, padding=True)
    optimizer_scheduler = get_optimizer(model, args)

    train_args = Seq2SeqTrainingArguments(
        evaluation_strategy="no",
        output_dir= args.paths.save_path,
        save_strategy="epoch",
        per_device_train_batch_size=args.train.batch_size,
        per_device_eval_batch_size=args.eval.batch_size,
        save_total_limit=2,
        num_train_epochs=args.train.epochs,
        predict_with_generate=True,
        # torch 2.0
        bf16=False,
        torch_compile=False,
        gradient_accumulation_steps=args.train.grad_acc_steps,
        logging_steps=args.logging.log_steps
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizers= optimizer_scheduler,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset = dataset["test"],
        data_collator=collate_fn,
    )

    trainer.train()



if __name__ == "__main__":
    train()

