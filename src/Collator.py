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

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        decoder_input_ids = [feature["decoder_input_ids"] for feature in features] if "decoder_input_ids" in features[0].keys() else None

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

        if decoder_input_ids is not None:
            max_dec_input_length = max(len(d) for d in decoder_input_ids)
            if self.pad_to_multiple_of is not None:
                max_dec_input_length = (
                    (max_dec_input_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                if labels is not None: 
                    label_remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                    if isinstance(feature["labels"], list):
                        feature["labels"] = (
                            feature["labels"] + label_remainder if padding_side == "right" else label_remainder + feature["labels"]
                        )
                    elif padding_side == "right":
                        feature["labels"] = np.concatenate([feature["labels"], label_remainder]).astype(np.int64)
                    else:
                        feature["labels"] = np.concatenate([label_remainder, feature["labels"]]).astype(np.int64)
                
                if decoder_input_ids is not None:
                    dec_input_remainder = [self.tokenizer.pad_token_id] * (max_dec_input_length - len(feature["decoder_input_ids"]))
                    if isinstance(feature["decoder_input_ids"], list):
                        feature["decoder_input_ids"] = (
                            feature["decoder_input_ids"] + dec_input_remainder if padding_side == "right" else dec_input_remainder + feature["decoder_input_ids"]
                        )
                    elif padding_side == "right":
                        feature["decoder_input_ids"] = np.concatenate([feature["decoder_input_ids"], dec_input_remainder]).astype(np.int64)
                    else:
                        feature["decoder_input_ids"] = np.concatenate([feature["decoder_input_ids"], dec_input_remainder]).astype(np.int64)

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
            input_ids = features["decoder_input_ids"]
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = self.model.config.decoder_start_token_id
            features["decoder_input_ids"] = shifted_input_ids
        return features