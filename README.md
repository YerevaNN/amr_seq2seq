# amr_seq2seq
AMR parsing with Seq2Seq architecture.
Based on AllenNLP framework built on top the PyTorch.

## Requirements

Only AllenNLP (version 0.7.2 or higher) and SentencePiece (0.1.8 or higher) are needed.

## Using

Training, Evaluation and Inference are done with standard AllenNLP command line
interface. For more detailed arguments please look at AllenNLP CLI documentaion.

### Training

To train a AMR parsing model you need to specify training configuration.
You can find examples of configuration files in `experiments/` directory.
No pre-processing step is needed.

For example:
```bash
allennlp train --include-package amr_seq2seq experiments/CONFIG.jsonnet -s SAVE_DIR
```

### Evaluation
In order to evaluate the model on a dataset:
```bash
allennlp evaluate --include-package amr_seq2seq --weights-file SAVE_DIR/best.th SAVE_DIR datasets/abstract_meaning_representation_amr_2.0/data/amrs/split/dev_all.txt --cuda-device 0
```

### Inference
To predict on a dataset with postprocessing:
```bash
allennlp predict --include-package amr_seq2seq --use-dataset-reader --silent --predictor noord_postprocessing --weights-file SAVE_DIR/best.th SAVE_DIR/ datasets/abstract_meaning_representation_amr_2.0/data/amrs/split/dev_all.txt --output-file OUTPUT_FILE.txt --cuda-device 0 --batch-size 150
```
