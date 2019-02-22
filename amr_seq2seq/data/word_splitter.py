from typing import List

from overrides import overrides

from allennlp.data import Token
from allennlp.data.tokenizers.word_splitter import WordSplitter


@WordSplitter.register('single_token')
class SingleTokenSplitter(WordSplitter):
    """
    Text splitting into one single token. Useful if token splitting
    is done in TokenIndexer instead. E.g. SentencePiece
    """

    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        single_token = Token(sentence)
        return [single_token]


@WordSplitter.register('noord_superchar')
class NoordSupercharSplitter(WordSplitter):
    """
    Superchar splitting based on Noord and Bos (2017) scripts.
    """
    def __init__(self,
                 model_path: str = None,
                 namespace: str = None):
        self.model_path = model_path
        self.namespace = namespace

    @overrides
    def split_words(self, sentence: str) -> List[Token]:

        amr = sentence.replace(' ', '+')  # replace actual spaces with '+'

        new_l = ''
        add_space = True
        for idx, ch in enumerate(amr):
            if ch == ':' and amr[idx + 1].isalpha():
                # after ':' there should always be a letter,
                # otherwise it is some URL probably and we just continue
                add_space = False
                new_l += ' ' + ch
            elif ch == '+':
                add_space = True
                new_l += ' ' + ch
            else:
                if add_space:
                    new_l += ' ' + ch
                else:  # we previously saw a ':', so we do not add spaces
                    new_l += ch

        spl = new_l.split()
        for idx, item in enumerate(spl):
            if len(item) > 1 and item[0] == ':':
                if any(x in item for x in
                       [')', '<', ')', '>', '/', 'jwf9X']):  # filter out non-structure words, due to links etc
                    new_str = ''
                    for ch in item:
                        new_str += ch + ' '
                    spl[idx] = new_str.strip()

        tokens = [Token(word) for word in spl]
        return tokens
