from typing import List, Dict

from overrides import overrides

from allennlp.common.util import pad_sequence_to_length

from allennlp.data import TokenType, Token
from allennlp.data.token_indexers import TokenIndexer

from allennlp.common.util import START_SYMBOL, END_SYMBOL

from .vocabulary import SubwordVocabulary


@TokenIndexer.register('subword')
class SubwordIndexer(TokenIndexer[int]):

    def __init__(self,
                 namespace: str = None,
                 model_path: str = None):
        self.namespace = namespace

        self.model_path = model_path
        self.model = None

        # self.bos_symbol = START_SYMBOL
        # self.eos_symbol = END_SYMBOL
        # self.bos_id = 2
        # self.eos_id = 3

        # if os.path.isdir(model_path) and os.listdir(model_path):
        # if os.path.exists(model_path):
        #     self.model = SentencePieceProcessor()
        #     self.model.load(model_path)
        #     assert(self.model.bos_id() == self.bos_id)
        #     assert(self.model.eos_id() == self.eos_id)

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # TODO check text_id case again
        text = token.text
        # counter[self.namespace][text] += 1  # TODO add
        counter[f'{self.namespace}-raw'][text] += 1

    # def split_text_into_subword_ids(self,
    #                                 text: str,
    #                                 vocabulary: Vocabulary) -> List[int]:
    #     # TODO vocabulary is temporary
    #     # TODO split with spm module w.r.t. given model
    #     subword_ids = self.model.encode_as_ids(text)
    #     # subword_ids = [vocabulary.get_token_index(text, self.namespace)]
    #     return subword_ids

    @overrides
    def tokens_to_indices(self, tokens: List[Token],
                          vocabulary: SubwordVocabulary,
                          index_name: str) -> Dict[str, List[int]]:
        # TODO load from vocabulary, not with __init__
        output_dict = {}

        # TODO handle BOS and EOS our own
        # raw_indices: List[int] = []
        #
        # raw_indices.append(self.bos_id)
        # for token in itertools.chain(tokens):
        #     word = token.text
        #     word_id = vocabulary.get_token_index(word, f'{self.namespace}-raw')
        #     raw_indices.append(word_id)
        # raw_indices.append(self.eos_id)
        #
        # output_dict[f'{index_name}-raw'] = raw_indices

        assert isinstance(vocabulary, SubwordVocabulary)
        if not self.model:
            self.model = vocabulary.get_subword_model(self.namespace)
            # assert self.model.bos_id() == self.bos_id
            # assert self.model.eos_id() == self.eos_id

        indices: List[int] = []

        # indices.append(self.bos_id)
        for token in tokens:
            word = token.text
            # check if word is already a piece / control char
            piece_id = self.model.piece_to_id(word)
            if self.model.is_control(piece_id):
                subword_ids = [piece_id]
            else:
                # subword_ids = self.model.encode_as_ids(word)
                subword_ids = self.model.sample_encode_as_ids(word, -1, 0.1)
            indices.extend(subword_ids)
        # indices.append(self.eos_id)

        output_dict[index_name] = indices

        output_dict['mask'] = [1 for _ in indices]
        return output_dict

        # if not self.model:
        #
        # # first, somehow split into subwords
        # pass

    @overrides
    def get_padding_token(self) -> int:
        return 0  # TODO

    @overrides
    def get_padding_lengths(self, token: TokenType) -> Dict[str, int]:
        return {}  # TODO

    @overrides
    def pad_token_sequence(self,
                           tokens: Dict[str, List[int]],
                           desired_num_tokens: Dict[str, int],
                           padding_lengths: Dict[str, int]) -> Dict[str, List[int]]:  # pylint: disable=unused-argument
        return {key: pad_sequence_to_length(val, desired_num_tokens[key])
                for key, val in tokens.items()}

    @overrides
    def get_keys(self, index_name: str) -> List[str]:
        return super().get_keys(index_name)
