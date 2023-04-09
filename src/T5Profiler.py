from deepspeed.profiling.flops_profiler import get_model_profile
import torch

class T5Profiler:
  # https://www.deepspeed.ai/tutorials/flops-profiler/#example-bert
  def __init__(self, model, tokenizer):
    self.model = model.to("cuda")
    self.tokenizer = tokenizer

  def __call__(self, batch_size, encoder_seq_len, decoder_seq_len = 1,kv_seq_len = 0, verbose = False):
    input = self.t5_input_constructor(batch_size, encoder_seq_len, decoder_seq_len, kv_seq_len)
    flops, macs, params = get_model_profile(
        self.model,
        kwargs=input,
        print_profile=verbose,
        detailed=False,
    )
    return {"flops":flops,"macs":macs, "params":params}

  def t5_input_constructor(self,batch_size, encoder_seq_len, decoder_seq_len, kv_seq_len):
    enc_seq = ["".join([self.tokenizer.pad_token for i in range(encoder_seq_len)])] * batch_size
    dec_seq = ["".join([self.tokenizer.pad_token for i in range(decoder_seq_len)])] * batch_size

    enc_tokens = self.tokenizer(enc_seq, padding = True, truncation = True, return_tensors = "pt").to("cuda")
    dec_tokens = self.tokenizer(dec_seq, padding = True, truncation = True, return_tensors = "pt").to("cuda")
    enc_tokens.update({"decoder_input_ids": dec_tokens["input_ids"]})

    if kv_seq_len > 0:
      p_kv = self.past_kv_constructor(batch_size, kv_seq_len)
      enc_tokens.update({"past_key_values": p_kv})
    return enc_tokens

  def past_kv_constructor(self, batch_size,kv_seq_len):
    if self.model.config.is_encoder_decoder:
      num_kv = 4
    else:
      num_kv = 2
    num_layers= self.model.config.num_decoder_layers
    num_heads = self.model.config.num_heads
    head_dim = self.model.config.d_kv

    past_kv = tuple(num_layers * [tuple(num_kv* [torch.rand(batch_size, num_heads, kv_seq_len, head_dim).to("cuda")])])
    return past_kv