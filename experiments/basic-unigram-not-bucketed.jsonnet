# some constants

local PADDING_TOKEN = "@@PADDING@@";
local UNKNOWN_TOKEN = "@@UNKNOWN@@";

local START_TOKEN = "@start@";
local END_TOKEN = "@end@";

# model-specific constants

local SOURCE_FIELD = "snt";
local TARGET_FIELD = "amr_linearized";
local RAW_TARGET_FIELD = "raw_amr";

local SOURCE_NAMESPACE = "snt_tokens";
local TARGET_NAMESPACE = "linearized_superchar_tokens";

# data-specific constants

local DATA_DIR = "datasets/abstract_meaning_representation_amr_2.0/data/amrs/";

# hyperparams

local NUM_EPOCHS = 500;

local BATCH_SIZE = 20;
local NUM_ITERATIONS_PER_EPOCH = 2000;

local OPTIMIZER = "dense_sparse_adam";

local VOCAB_SIZE = 1000;

local EMBEDDING_DIM = 300;
local HIDDEN_DIM = 300;

local DROPOUT = 0.3;

local ENCODER_CELL = "lstm";
local NUM_ENCODER_LAYERS = 3;
local BIDIRECTIONAL = true;

local ENCODER_DIM = (if BIDIRECTIONAL then 2 * HIDDEN_DIM else HIDDEN_DIM);



# validation and inference hyperparams

local BEAM_SIZE = 4;
local VAL_BATCH_SIZE = 200;
local MAX_DECODING_STEPS = 500;

local SUMMARY_INTERVAL = 400;

#

local NUM_INSTANCES_PER_EPOCH = BATCH_SIZE * NUM_ITERATIONS_PER_EPOCH;


//local char_tokenizer = {
//    "type": "character",
//    "lowercase_characters": true
//};
//
//local superchar_tokenizer = {
//    "type": "word",
//    "word_splitter": {
//        "type": "noord_superchar"
//    },
//    "start_tokens": [START_TOKEN],
//    "end_tokens": [END_TOKEN]
//};


local single_token_tokenizer = {
  "type": "word",
  "word_splitter": {
    "type": "single_token"
  },
  "start_tokens": [START_TOKEN],
  "end_tokens": [END_TOKEN]
};


//local single_id_indexer(namespace, index="tokens") = {
//    [index]: {
//        "type": "single_id",
//        "namespace": namespace
//    }
//};

local subword_token_indexer(namespace, index="tokens") = {
    [index]: {
        "type": "subword",
        "namespace": namespace
    }
};



{
    "dataset_reader": {
        "type": "amr_reader",

        "snt_tokenizer": single_token_tokenizer,
        "snt_token_indexers": subword_token_indexer(SOURCE_NAMESPACE),

        "linearized_amr_tokenizer": single_token_tokenizer,
        "linearized_amr_indexers": subword_token_indexer(TARGET_NAMESPACE),

        "lazy": true
    },


    "iterator": {
        "type": "basic",
        "batch_size": BATCH_SIZE,
        "instances_per_epoch": NUM_INSTANCES_PER_EPOCH
    },
    "validation_iterator": {
        "type": "bucket",
        "batch_size": VAL_BATCH_SIZE,
        "biggest_batch_first": true,
        "sorting_keys": [
            [SOURCE_FIELD, "num_tokens"]
        ]
    },



    "vocabulary": {  # Vocab hyperparams are shared
        "type": "subword",
        "training_params": {
          "vocab_size": VOCAB_SIZE,
          "user_defined_symbols": ['(', ')']
        }
    },


    "model": {
        "type": "translation",

        "source_embedder": {
            "type": "basic",
            "allow_unmatched_keys": true,
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": EMBEDDING_DIM,
                    "vocab_namespace": SOURCE_NAMESPACE
                }
            }
        },

        "encoder": {
            "type": ENCODER_CELL,
            "bidirectional": BIDIRECTIONAL,
            "dropout": DROPOUT,
            "hidden_size": HIDDEN_DIM,
            "input_size": EMBEDDING_DIM,
            "num_layers": NUM_ENCODER_LAYERS
        },

        "attention": {
            "type": "bilinear",
            "matrix_dim": ENCODER_DIM,
            "vector_dim": ENCODER_DIM
        },

        "source_field": SOURCE_FIELD,
        "target_field": TARGET_FIELD,
        "raw_target_field": RAW_TARGET_FIELD,

        "target_namespace": TARGET_NAMESPACE,

        "max_decoding_steps": MAX_DECODING_STEPS,
        "beam_size": BEAM_SIZE
    },

    "train_data_path": DATA_DIR + "split/training_all.txt",
    "validation_data_path": DATA_DIR + "split/dev_all.txt",

    "trainer": {
        "cuda_device": 0,
        "num_epochs": NUM_EPOCHS,
        "num_serialized_models_to_keep": NUM_EPOCHS,
        "optimizer": {
            "type": OPTIMIZER
        },
        "summary_interval": SUMMARY_INTERVAL
    }
}