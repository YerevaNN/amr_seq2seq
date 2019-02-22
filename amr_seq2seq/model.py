from typing import List, Dict, Any

from overrides import overrides

import torch

from allennlp.data import Vocabulary
from allennlp.models import Model, SimpleSeq2Seq
from allennlp.modules import Seq2SeqEncoder
from allennlp.modules import TextFieldEmbedder
from allennlp.modules import Attention
from allennlp.training.metrics import Metric

from .metrics import Smatch
from .utils import postprocess_AMRs


@Model.register('translation')
class TranslationModel(SimpleSeq2Seq):
    """
    TODO TODO TODO
    TODO TODO
    TODO
    """
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 beam_size: int,
                 max_decoding_steps: int,
                 attention: Attention,
                 target_namespace: str,
                 source_field: str,
                 target_field: str,
                 raw_target_field: str = 'raw_amr',
                 use_bleu: bool = True):
        super().__init__(vocab=vocab,
                         source_embedder=source_embedder,
                         encoder=encoder,
                         max_decoding_steps=max_decoding_steps,
                         beam_size=beam_size,
                         target_namespace=target_namespace,
                         use_bleu=use_bleu,
                         attention=attention)

        self.source_field = source_field
        self.target_field = target_field
        self.raw_target_field = raw_target_field

        self._smatch: Metric = Smatch(restart_number=10)

    @overrides
    def forward(self,
                **inputs: Dict[str, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Make forward pass with decoder logic for producing the entire target sequence.
        The fields corresponding to arguments `source_field` and `target_field`
        are used as source and target.
        In the inference mode targets are used for computing metrics.
        """
        source_tokens = inputs.get(self.source_field)
        target_tokens = inputs.get(self.target_field, None)
        raw_target: List[str] = inputs.get(self.raw_target_field, None)

        # Call encode-decode process
        output_dict = super().forward(source_tokens, target_tokens)

        # During evaluation, when target is provided and the model is
        # running in inference mode, always do the decoding step to
        # enable metrics to use finalized predictions instead of tensors
        if raw_target and self._smatch and not self.training:
            output_dict = self.decode(output_dict)
            prediction = output_dict["predicted_amr"]

            self._smatch(prediction, raw_target)

        return output_dict

    def postprocess_predicted_text(self, batch_text: List[str]):
        """
        For finalizing predictions, postprocessing similar to
        Noord and Bos (2017) is done.
        """
        batch_amrs = []
        for text in batch_text:
            amr = postprocess_AMRs.process_item(text)
            batch_amrs.append(amr)
        return batch_amrs

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions. Tensors are converted back into tokens using
        the vocabulary. Tokens are then concatenated to get a linearized amrs.
        Finally, postprocessing is done to get valid amr representations.
        """

        # Prepare tokens
        output_dict = super().decode(output_dict)

        # Turn tokens into raw text
        batch_predicted_text = []
        for predicted_tokens in output_dict['predicted_tokens']:
            predicted_text = self.detokenize(predicted_tokens)
            batch_predicted_text.append(predicted_text)
        output_dict['predicted_text'] = batch_predicted_text

        # Postprocess predicted sequences
        predicted_amr = self.postprocess_predicted_text(batch_predicted_text)
        output_dict['predicted_amr'] = predicted_amr

        return output_dict

    def detokenize(self, tokens: List[str]) -> str:
        """
        Detokenize given lists of tokens. If the does not provide detokenization
        procedure, use the default one instead
        """
        if hasattr(self.vocab, 'detokenize'):
            return self.vocab.detokenize(tokens, namespace=self._target_namespace)
        return ''.join(tokens).replace('+', ' ').replace('   ', '+').replace('▁', ' ').strip()  # TODO

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        Get metrics for current state.
        """

        all_metrics: Dict[str, float] = super().get_metrics(reset=reset)

        if self._smatch and not self.training:
            all_metrics.update(self._smatch.get_metric(reset=reset))

        return all_metrics
