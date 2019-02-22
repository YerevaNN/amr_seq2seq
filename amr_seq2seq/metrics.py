from typing import Dict, List

from overrides import overrides

from allennlp.training.metrics import Metric

from .utils.smatch_edited import SmatchScript

import logging
logging.getLogger('amr_postprocessing').setLevel(logging.CRITICAL)


@Metric.register('smatch')
class Smatch(Metric):
    """
    Smatch metric for evaluating AMR predictions.
    Uses modified version of smatch evaluating script (SmatchScript).

    This Metric allows to observe metric during the training process.
    Very useful for model selection, early stopping and visualizing.
    """
    def __init__(self,
                 restart_number: int = 4,
                 just_instance: bool = False,
                 just_attribute: bool = False,
                 just_relation: bool = False):
        """
        Look original smatch evaluation script for reference.
        """
        self.restart_number = restart_number
        self.just_instance = just_instance
        self.just_attribute = just_attribute
        self.just_relation = just_relation

        # As metric state we use SmatchScript object
        self.state: SmatchScript = None
        self.reset()

    @overrides
    def __call__(self,
                 predictions: List[str],
                 gold_labels: List[str]) -> None:
        """
        Accumulate statistics on batch of predictions and targets.
        """
        for prediction, gold_label in zip(predictions, gold_labels):
            self.state.process_instance(prediction, gold_label)

    @overrides
    def get_metric(self, reset: bool) -> Dict[str, float]:
        """
        Calculate final metrics score out of accumulated statistics.
        """
        metrics_dict = self.state.get_metrics()
        if reset:
            self.reset()
        return metrics_dict

    @overrides
    def reset(self) -> None:
        """
        Prepare new state instead of manually resetting statistics.
        """
        self.state = SmatchScript(r=self.restart_number,
                                  justinstance=self.just_instance,
                                  justattribute=self.just_attribute,
                                  justrelation=self.just_relation)
