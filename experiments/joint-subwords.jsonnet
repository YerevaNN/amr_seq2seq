local joint_subword_namespace = "joint_subword_namespace";
local snt_namespace = joint_subword_namespace;
local amr_namespace = joint_subword_namespace;

local E = 100;
local H = 300;
local N = 3;
local D = 0;
local V = 3000;
local MAX_T = 1200;
local BEAM_SIZE = 8;
local BATCH_SIZE = 100;
local NUM_EPOCHS = 500;

local subword_tokenizer = {
  "type": "word",
  "word_splitter": {
    "type": "single_token"
  },
  "start_tokens": ["<s>"],
  "end_tokens": ["</s>"]
};

local snt_token_indexers = {
  "tokens": {
    "type": "subword",
    "namespace": snt_namespace
  }
};

local linearized_amr_indexers = {
  "tokens": {
    "type": "subword",
    "namespace": amr_namespace
  }
};

{
  "dataset_reader": {
    "type": "amr_reader",
    "snt_tokenizer": subword_tokenizer,
    "linearized_amr_tokenizer": subword_tokenizer,
    "snt_token_indexers": snt_token_indexers,
    "linearized_amr_indexers": linearized_amr_indexers,
    "lazy": true,
    "graph": false
  },
  "vocabulary": {
    "type": "subword",
//    "directory_path": "joint/vocabulary"
    "training_params": {
      "vocab_size": V,
      "user_defined_symbols": ['(', ')', '-', '/', ':']
    }
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
          "vocab_namespace": snt_namespace,
          "embedding_dim": E,
          "trainable": true
        }
      },
      "allow_unmatched_keys": true
    },
    "encoder": {
      "type": "lstm",
      "input_size": E,
      "hidden_size": H,
      "num_layers": N,
      "dropout": D,
      "bidirectional": true
    },
    "max_decoding_steps": MAX_T,
    "target_namespace": amr_namespace,
    "attention": {
      "type": "bilinear",
      "vector_dim": 2 * H,
      "matrix_dim": 2 * H
    },
    "beam_size": BEAM_SIZE,
    "use_bleu": true,
    "source_field": "snt",
    "target_field": "amr_linearized",
    "raw_target_field": "raw_amr"
  },
  "iterator": {
    "type": "bucket",
    "padding_noise": 0,
    "batch_size": BATCH_SIZE,
    "max_instances_in_memory": 100000,
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
    "num_epochs": NUM_EPOCHS,
    "summary_interval": 100,
    "cuda_device": 0,
    "num_serialized_models_to_keep": 500,
    "optimizer": {
      "type": "adam"//,
//      "lr": 0.05,
//      "momentum": 0.9,
//      "nesterov": true
    }
  }
}