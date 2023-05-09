import numpy as np
from typing import Any, Optional, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from dataclasses import dataclass

@dataclass
class DataCollatorForPromptedSeq2Seq:
    """
    Adjusted version from HF DatacollatorForSeq2Seq
    - Added dynamic padding for decoder_input_ids
    - Pads input_ids and labels (+decoder_input_ids) to the same length
    - Shifts decoder_input_ids to the right
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def get_max_input_length(self, feature):
        max_label_length = max(len(l) for l in feature)
        if self.pad_to_multiple_of is not None:
            max_label_length = (
                (max_label_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )
        return max_label_length

    def apply_padding(self,feature, max_label_length, padding_side, is_label=False):
        pad_token = self.tokenizer.pad_token_id if not is_label else self.label_pad_token_id
        label_remainder = [pad_token] * (max_label_length - len(feature))
        if isinstance(feature, list):
            feature = (
                feature + label_remainder if padding_side == "right" else label_remainder + feature
            )
        elif padding_side == "right":
            feature = np.concatenate([feature, label_remainder]).astype(np.int64)
        else:
            feature = np.concatenate([label_remainder, feature]).astype(np.int64)
        attn_mask = np.ones_like(feature)
        attn_mask[-len(label_remainder):] = 0
        return feature, attn_mask

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        decoder_input_ids = [feature["decoder_input_ids"] for feature in features] if "decoder_input_ids" in features[0].keys() else None
        p_input_ids = [feature["p_input_ids"] for feature in features] if "p_input_ids" in features[0].keys() else None


        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = self.get_max_input_length(labels)

        if decoder_input_ids is not None:
            max_dec_input_length = self.get_max_input_length(decoder_input_ids)
        
        if p_input_ids is not None:
            max_p_input__length = self.get_max_input_length(p_input_ids)

        padding_side = self.tokenizer.padding_side
        for feature in features:
            if labels is not None: 
                feature["labels"], _ = self.apply_padding(feature["labels"], max_label_length, padding_side, is_label=True)

            if decoder_input_ids is not None:
                feature["decoder_input_ids"], feature["decoder_attention_mask"] = self.apply_padding(feature["decoder_input_ids"], max_dec_input_length, padding_side)
    
            if p_input_ids is not None:
                feature["p_input_ids"], feature["p_attention_mask"] = self.apply_padding(feature["p_input_ids"], max_p_input__length, padding_side)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and decoder_input_ids is None # added
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids
        elif (decoder_input_ids is not None
             and self.model is not None
        ):
            # Shifts ids to the right and ads decoder start token
            input_ids = features["decoder_input_ids"]
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = self.model.config.decoder_start_token_id
            features["decoder_input_ids"] = shifted_input_ids
        return features