local amr_namespace = "linearized_superchar_tokens";

{
  "dataset_reader": {
    "type": "amr_reader",
    "snt_tokenizer": {
      "type": "character"
    },
    "linearized_amr_tokenizer": {
      "type": "word",
      "word_splitter": {
        "type": "just_spaces"
      },
      "start_tokens": ["<s>"],
      "end_tokens": ["</s>"]
    },
    "snt_token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "snt_tokens"
      }
    },
    "linearized_amr_indexers": {
      "word_tokens": {
        "type": "single_id",
        "namespace": "amr_raw_REMOVE_OR_NOT"
      },
      "tokens": {
        "type": "subword",
        "model_path": $["vocabulary"]["directory_path"] + '/.models/' + amr_namespace + '.model',
        "namespace": amr_namespace
      }
    },
    "lazy": false,
    "graph": false
  },
  "vocabulary": {
    "type": "subword",
    "directory_path": "fake/vocabulary"
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
    "target_namespace": amr_namespace,
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
    "batch_size": 50,
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
    "biggest_batch_first": true
  },
  "trainer": {
    "num_epochs": 500,
    "summary_interval": 100,
    "cuda_device": 0,
    "num_serialized_models_to_keep": 200,
    "optimizer": {
      "type": "adam"//,
//      "lr": 0.05,
//      "momentum": 0.9,
//      "nesterov": true
    }
  }
}