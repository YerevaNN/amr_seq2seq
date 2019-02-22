from typing import Iterable, Dict, Union, Optional, List, Any

import os
import math
import codecs
import shutil
import logging
import tempfile

from collections import defaultdict

from allennlp.common import Params
from allennlp.common.tqdm import Tqdm
from allennlp.data import vocabulary
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN

from allennlp.common.util import START_SYMBOL, END_SYMBOL

from sentencepiece import SentencePieceTrainer
from sentencepiece import SentencePieceProcessor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

MODELS_DIR = '.models'


@Vocabulary.register('subword')
class SubwordVocabulary(Vocabulary):
    """
    Vocabulary implementation for AllenNLP SentencePiece integration.
    """

    def get_token_index(self, token: str, namespace: str = 'tokens') -> int:
        # if token == START_SYMBOL:
        #     token = '<s>'
        # if token == END_SYMBOL:
        #     token = '</s>'
        return super().get_token_index(token, namespace)

    def __init__(self,
                 namespace_model_temp_dir: tempfile.TemporaryDirectory = None,
                 namespace_model_paths: Dict[str, str] = None,
                 counter: Dict[str, Dict[str, int]] = None, min_count: Dict[str, int] = None,
                 max_vocab_size: Union[int, Dict[str, int]] = None,
                 non_padded_namespaces: Iterable[str] = vocabulary.DEFAULT_NON_PADDED_NAMESPACES,
                 pretrained_files: Optional[Dict[str, str]] = None, only_include_pretrained_words: bool = False,
                 tokens_to_add: Dict[str, List[str]] = None, min_pretrained_embeddings: Dict[str, int] = None) -> None:
        super().__init__(counter, min_count, max_vocab_size, non_padded_namespaces, pretrained_files,
                         only_include_pretrained_words, tokens_to_add, min_pretrained_embeddings)
        self._namespace_model_temp_dir = namespace_model_temp_dir
        self._namespace_model_paths = namespace_model_paths or {}
        # TODO load all with init

    def get_subword_model(self, namespace: str) -> SentencePieceProcessor:
        model_path = self._namespace_model_paths[namespace]

        model = SentencePieceProcessor()
        model.load(model_path)

        return model

    def save_to_files(self, directory: str) -> None:
        super().save_to_files(directory)
        temp_models_dir = os.path.join(self._namespace_model_temp_dir.name, MODELS_DIR)
        models_dir = os.path.join(directory, MODELS_DIR)

        shutil.copytree(temp_models_dir,
                        models_dir)

        # os.makedirs(model_dir, exist_ok=True)
        # for filename in os.listdir(self._namespace_model_temp_dir.name):
        #     shutil.copyfile(filename)
        # for namespace, path in self._namespace_model_paths.items():
        #     model_name = f'{namespace}.model'
        #     os.cp(path, os.path.join(model_dir, model_name))

    @classmethod
    def from_files(cls, directory: str,
                   training_params: Dict[str, Any] = None) -> 'SubwordVocabulary':
        vocab: 'SubwordVocabulary' = super().from_files(directory)
        # Check every file in the directory.

        namespace_model_temp_dir = tempfile.TemporaryDirectory()
        temp_models_dir = os.path.join(namespace_model_temp_dir.name, MODELS_DIR)
        models_dir = os.path.join(directory, MODELS_DIR)
        shutil.copytree(models_dir, temp_models_dir)

        for namespace_filename in os.listdir(temp_models_dir):
            if namespace_filename.startswith('.'):
                continue
            if not namespace_filename.endswith('.model'):
                continue
            filename = os.path.join(temp_models_dir, namespace_filename)
            namespace = namespace_filename.replace('.model', '')
            vocab._namespace_model_paths[namespace] = filename

        vocab._namespace_model_temp_dir = namespace_model_temp_dir

        return vocab

    def set_from_file(self, filename: str, is_padded: bool = True, oov_token: str = vocabulary.DEFAULT_OOV_TOKEN,
                      namespace: str = "tokens"):
        super().set_from_file(filename, is_padded, oov_token, namespace)

    @classmethod
    def from_instances(cls, instances: Iterable['adi.Instance'],
                       training_params: Dict[str, Any] = None,
                       min_count: Dict[str, int] = None,
                       max_vocab_size: Union[int, Dict[str, int]] = None,
                       non_padded_namespaces: Iterable[str] = vocabulary.DEFAULT_NON_PADDED_NAMESPACES,
                       pretrained_files: Optional[Dict[str, str]] = None, only_include_pretrained_words: bool = False,
                       tokens_to_add: Dict[str, List[str]] = None,
                       min_pretrained_embeddings: Dict[str, int] = None):
        logger.info("Fitting token dictionary from dataset.")

        if not training_params:
            training_params = {}

        namespace_token_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for instance in Tqdm.tqdm(instances):
            instance.count_vocab_items(namespace_token_counts)

            # for field in instance.fields.values():
            #     if not isinstance(field, TextField):
            #         continue
            #     field: TextField
            #
            #
            #     for indexer in field._token_indexers.values():
            #         for token in field.tokens:
            #             indexer.count_vocab_items(token, counter)

        # if we are building vocabulary from instances, we need to do
        # training of our subword model

        namespace_token_approx_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        namespace_model_paths = {}
        namespace_model_temp_dir = tempfile.TemporaryDirectory()

        for counter, token_counts in namespace_token_counts.items():
            namespace, _, tail = counter.rpartition('-raw')
            if tail:
                continue
            # del namespace_token_counts[counter]

            assert(namespace not in namespace_token_counts)

            with tempfile.NamedTemporaryFile('w', encoding='utf-8') as f:
                logger.info(f'Preparing statistics for namespace {namespace}')
                num_tokens = 0
                for token, counts in Tqdm.tqdm(token_counts.items()):
                    if token == START_SYMBOL or token == END_SYMBOL:
                        continue
                    for _ in range(counts):
                        f.write(token)
                        f.write(os.linesep)
                    num_tokens += counts * len(token.split())
                f.flush()

                filename = f.name
                model_dir = os.path.join(namespace_model_temp_dir.name, MODELS_DIR)
                os.makedirs(model_dir, exist_ok=True)
                model_prefix = os.path.join(model_dir, namespace)

                model_path = f'{model_prefix}.model'
                vocab_path = f'{model_prefix}.vocab'

                # TODO bos_piece is added, use that feature

                training_params = {
                    'pad_id': 0,
                    'pad_piece': DEFAULT_PADDING_TOKEN,
                    'unk_id': 1,
                    'unk_piece': DEFAULT_OOV_TOKEN,
                    'unk_surface': DEFAULT_OOV_TOKEN,
                    'bos_id': 2,
                    'bos_piece': START_SYMBOL,
                    'eos_id': 3,
                    'eos_piece': END_SYMBOL,
                    'model_type': 'unigram',
                    'user_defined_symbols': [],
                    **training_params,
                    'input': filename,
                    'model_prefix': model_prefix
                }

                assert 'vocab_size' in training_params

                train_args = []
                for key, val in training_params.items():
                    if isinstance(val, list):
                        val = ','.join(val)
                    train_args.append(f'--{key}={val}')
                train_args = ' '.join(train_args)

                SentencePieceTrainer.Train(train_args)

                namespace_model_paths[namespace] = model_path
                # TODO make all files temporary

                with codecs.open(vocab_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        subword, _, log_probability = line.strip().partition('\t')
                        if subword in [DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN]:
                            continue
                        log_probability = float(log_probability)
                        approx_num_occurences = num_tokens * math.exp(log_probability)
                        approx_num_occurences = math.ceil(approx_num_occurences)
                        namespace_token_approx_counts[namespace][subword] = approx_num_occurences

                print('pkay')

        namespace_token_counts.update(namespace_token_approx_counts)

        return cls(namespace_model_temp_dir=namespace_model_temp_dir,
                   namespace_model_paths=namespace_model_paths,
                   counter=namespace_token_counts,
                   min_count=min_count,
                   max_vocab_size=max_vocab_size,
                   non_padded_namespaces=non_padded_namespaces,
                   pretrained_files=pretrained_files,
                   only_include_pretrained_words=only_include_pretrained_words,
                   tokens_to_add=tokens_to_add,
                   min_pretrained_embeddings=min_pretrained_embeddings)

    @classmethod
    def from_params(cls, params: Params, instances: Iterable['adi.Instance'] = None):  # type: ignore

        path = params.get('directory_path', '')
        if not os.path.isdir(path) or not os.listdir(path):
            params.pop('directory_path', '')

        # and the rest ...

        vocab_type = params.pop("type", None)
        if vocab_type is not None:
            return cls.by_name(vocab_type).from_params(params=params, instances=instances)

        extend = params.pop("extend", False)
        vocabulary_directory = params.pop("directory_path", None)
        if not vocabulary_directory and not instances:
            raise vocabulary.ConfigurationError("You must provide either a Params object containing a "
                                     "vocab_directory key or a Dataset to build a vocabulary from.")
        if extend and not instances:
            raise vocabulary.ConfigurationError("'extend' is true but there are not instances passed to extend.")
        if extend and not vocabulary_directory:
            raise vocabulary.ConfigurationError("'extend' is true but there is not 'directory_path' to extend from.")

        if vocabulary_directory and instances:
            if extend:
                logger.info("Loading Vocab from files and extending it with dataset.")
            else:
                logger.info("Loading Vocab from files instead of dataset.")

        if vocabulary_directory:
            vocab = cls.from_files(vocabulary_directory)
            if not extend:
                params.assert_empty("Vocabulary - from files")
                return vocab
        if extend:
            vocab.extend_from_instances(params, instances=instances)
            return vocab
        training_params = params.pop("training_params", {})
        min_count = params.pop("min_count", None)
        max_vocab_size = vocabulary.pop_max_vocab_size(params)
        non_padded_namespaces = params.pop("non_padded_namespaces", vocabulary.DEFAULT_NON_PADDED_NAMESPACES)
        pretrained_files = params.pop("pretrained_files", {})
        min_pretrained_embeddings = params.pop("min_pretrained_embeddings", None)
        only_include_pretrained_words = params.pop_bool("only_include_pretrained_words", False)
        tokens_to_add = params.pop("tokens_to_add", None)
        params.assert_empty("Vocabulary - from dataset")
        return cls.from_instances(instances=instances,
                                  training_params=training_params,
                                  min_count=min_count,
                                  max_vocab_size=max_vocab_size,
                                  non_padded_namespaces=non_padded_namespaces,
                                  pretrained_files=pretrained_files,
                                  only_include_pretrained_words=only_include_pretrained_words,
                                  tokens_to_add=tokens_to_add,
                                  min_pretrained_embeddings=min_pretrained_embeddings)
