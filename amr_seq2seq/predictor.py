from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.predictors import Predictor


@Predictor.register('translation')
class TranslationPredictor(Predictor):
    """
    Simple predictor returning raw linearized amrs.
    """

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        # Some steps are may seem redundant as they are
        # here for backward compatibility with out previously trained models
        predicted_tokens = outputs['predicted_tokens']
        predicted_str = ''.join(predicted_tokens)
        predicted_str = predicted_str.replace('+', ' ').replace('   ', ' + ').replace('▁', ' ')
        predicted_str, _, _ = predicted_str.partition('</s>')
        predicted_str = predicted_str.strip()
        return predicted_str + '\n'


@Predictor.register('noord_postprocessing')
class NoordPostprocessingPredictor(Predictor):
    """
    Predictor that return linearized amrs after postprocessing.
    """

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        predicted_str = outputs['predicted_amr']
        return predicted_str + '\n'
