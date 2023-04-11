from utils.utils import load_args
from utils.training_utils import get_optimizer
from data_scripts.Collator import DataCollatorForPromptedSeq2Seq

from datasets import load_from_disk
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer, T5ForConditionalGeneration
import wandb

def train():
    args = load_args()
    if hasattr(args, 'wandb_key'):
        wandb.login(key=args.wandb_key)
    dataset = load_from_disk(args.paths.data_path)
    if args.optim.total_steps is None: args.optim.total_steps = len(dataset["train"]) / args.train.grad_acc_steps * args.train.epochs
    model = T5ForConditionalGeneration.from_pretrained(args.model.path).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model.path)
    tokenizer.add_special_tokens({'sep_token': args.model.sep_token})
    model.resize_token_embeddings(len(tokenizer))
    collate_fn = DataCollatorForPromptedSeq2Seq(tokenizer, model, padding=True)
    optimizer_scheduler = get_optimizer(model, args)

    train_args = Seq2SeqTrainingArguments(
        evaluation_strategy="no",
        output_dir= args.paths.save_path,
        save_strategy="steps",
        save_steps=args.train.save_steps,
        per_device_train_batch_size=args.train.batch_size,
        per_device_eval_batch_size=args.eval.batch_size,
        save_total_limit=2,
        num_train_epochs=args.train.epochs,
        predict_with_generate=True,
        # torch 2.0
        bf16=False,
        torch_compile=False,
        gradient_accumulation_steps=args.train.grad_acc_steps,
        logging_steps=args.train.log_steps
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
    trainer.save_model(args.paths.save_path)



if __name__ == "__main__":
    train()

