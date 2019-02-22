local BATCH_SIZE = 20;
local VAL_BATCH_SIZE = 200;
local NUM_ITERATIONS_PER_EPOCH = 500;

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
  "train_data_path": "datasets/abstract_meaning_representation_amr_2.0/data/amrs/split/training_all.txt",
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
      "dropout": 0.7,
      "bidirectional": true
    },
    "max_decoding_steps": 500,
    "target_namespace": "linearized_superchar_tokens",
    "attention": {
      "type": "bilinear",
      "vector_dim": 600,
      "matrix_dim": 600
    },
    "beam_size": 4,
    "use_bleu": true,
    "source_field": "snt",
    "target_field": "amr_linearized",
    "raw_target_field": "raw_amr"
  },
  "iterator": {
    "type": "basic",
    "instances_per_epoch": BATCH_SIZE * NUM_ITERATIONS_PER_EPOCH,
    "batch_size": BATCH_SIZE,
    "cache_instances": false
  },
  "validation_iterator": {
    "type": "bucket",
    "batch_size": VAL_BATCH_SIZE,
    "sorting_keys": [
      ["snt", "num_tokens"]
    ],
    "max_instances_in_memory": 100000,
    "biggest_batch_first": true
  },
  "trainer": {
    "num_epochs": 5000,
    "summary_interval": 100,
    "cuda_device": 0,
    "num_serialized_models_to_keep": 2000,
    "optimizer": {
      "type": "adam"
    }
  }
}