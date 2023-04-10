from torch.optim.lr_scheduler import SequentialLR,LinearLR,CosineAnnealingLR
from torch.optim import AdamW
from transformers import Adafactor
import torch 


# Adjusted code from: https://github.com/PiotrNawrot/nanoT5/blob/main/nanoT5/utils/model_utils.py
def get_optimizer(model, args):
    no_decay = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.optim.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    assert args.optim.type in ["adam", "adafactor"], "Optimizer type not supported (adam, adafactor)"
    if args.optim.type == "adam":
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.optim.lr,fused=True if torch.__version__ >= "2.0.0" else False)
    elif args.optim.type == "adafactor":
        optimizer = Adafactor(optimizer_grouped_parameters, lr=args.optim.lr, scale_parameter=False, relative_step=False)


    scheduler1 = LinearLR(
        optimizer,
        start_factor=0.5,
        end_factor=1,
        total_iters=args.optim.warmup_steps,
        last_epoch=-1,
    )

    scheduler2 = CosineAnnealingLR(
        optimizer,
        T_max=args.optim.total_steps - args.optim.warmup_steps,
        eta_min=args.optim.final_cosine,
    )

    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler1, scheduler2],
        milestones=[args.optim.warmup_steps]
    )
    return optimizer, lr_scheduler