local BATCH_SIZE = 30;
local NUM_ITERATIONS_PER_EPOCH = 3500;

{
  "dataset_reader": {
    "type": "amr_reader",
    "snt_tokenizer": {
      "type": "character",
      "lowercase_characters": false,
      "start_tokens": ["@start@"],
      "end_tokens": ["@end@"]
    },
    "snt_token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "snt_tokens"
      }
    },
    "linearized_amr_tokenizer": {
      "type": "word",
      "word_splitter": {
        "type": "noord_superchar"
      },
      "start_tokens": ["@start@"],
      "end_tokens": ["@end@"]
    },
    "lazy": false,
    "graph": false
  },
  "train_data_path": "datasets/abstract_meaning_representation_amr_2.0/data/amrs/split/dev_all.txt",
  "validation_data_path": "datasets/abstract_meaning_representation_amr_2.0/data/amrs/split/dev_all.txt",
  "model": {
    "type": "translation",
    "source_embedder": {
      "type": "basic",
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "vocab_namespace": "snt_tokens",
          "embedding_dim": 300,
          "trainable": true
        }
      },
      "allow_unmatched_keys": true
    },
    "encoder": {
      "type": "lstm",
      "input_size": 300,
      "hidden_size": 300,
      "num_layers": 3,
      "dropout": 0,
      "bidirectional": true
    },
    "max_decoding_steps": 500,
    "target_namespace": "linearized_superchar_tokens",
    "attention": {
      "type": "bilinear",
      "vector_dim": 600,
      "matrix_dim": 600
    },
    "beam_size": 8,
    "use_bleu": true,
    "source_field": "snt",
    "target_field": "amr_linearized"
  },
  "iterator": {
    "type": "bucket",
    "padding_noise": 0,
    "batch_size": BATCH_SIZE,
//    "max_instances_in_memory": 100000,
    "sorting_keys": [
      [
        "snt",
        "num_tokens"
      ],
      [
        "amr_linearized",
        "num_tokens"
      ]
    ],
    "biggest_batch_first": true,
    "instances_per_epoch": BATCH_SIZE * NUM_ITERATIONS_PER_EPOCH
  },
  "trainer": {
    "num_epochs": 5000,
    "summary_interval": 20,
    "cuda_device": 0,
    "num_serialized_models_to_keep": 1000,
    "validation_metric": "+SMATCH",
    "optimizer": {
      "type": "adam"//,
//      "lr": 0.05,
//      "momentum": 0.9,
//      "nesterov": true
    }
  }
}
