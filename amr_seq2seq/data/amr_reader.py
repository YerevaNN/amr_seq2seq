from typing import Iterator, List, Dict, Tuple, TextIO
from typing import overload

import os
import io
import re
import zlib
import penman
import logging

from glob import glob
from overrides import overrides

from allennlp.data import Instance, DatasetReader
from allennlp.data.fields import Field, TextField, MetadataField

from allennlp.data.tokenizers import Tokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from .amr_graph_field import AMRGraphField

Line = str
Lines = List[str]


@DatasetReader.register('amr_reader')
class AMRReader(DatasetReader):
    """
    DatasetReader for Abstract Meaning Representations given in PENMAN notation.
    Alignements are not supported (yet).
    Supports lazy mode.
    """
    def __init__(self, *,
                 snt_tokenizer: Tokenizer,
                 linearized_amr_tokenizer: Tokenizer,
                 snt_token_indexers: Dict[str, TokenIndexer],
                 linearized_amr_indexers: Dict[str, TokenIndexer],
                 lazy: bool = False,
                 graph: bool = False
                 ):
        self.graph = graph
        self.amr_codec = penman.AMRCodec()

        self.snt_tokenizer = snt_tokenizer
        self.snt_token_indexers = snt_token_indexers
        self.linearized_amr_tokenizer = linearized_amr_tokenizer

        if not linearized_amr_indexers:
            # legacy support
            linearized_amr_indexers = {
                "tokens": SingleIdTokenIndexer(namespace="snt_tokens")
            }
        self.linearized_amr_indexers = linearized_amr_indexers

        super().__init__(lazy=lazy)

    @classmethod
    def decode_metadata(cls, lines: str) -> Dict[str, str]:
        """
        Decode metadata in standard AMR format.
        E.g.
            # ::id bolt12_10495_0392.5 ::date 2012-12-12T20:30:09 ::annotator SDL-AMR-09 ::preferred
            # ::snt Here it is a country with the freedom of speech.
            # ::save-date Tue Apr 28, 2015 ::file bolt12_10495_0392_5.txt
        """
        pattern = re.compile(r'''
            (?: ::)             # key always starts with ::
            (   \S+)            # capture non-whitespace name of the key 
            (?: \s*?)           # whitespace separator
            (   \S  .*?)?       # capture [optional] value starts with non-whitespace
            (?= \s+ ::\S+ | $)  # until next `::key` or `EOF`
        ''', re.DOTALL | re.VERBOSE | re.UNICODE)
        matches = pattern.findall(lines)
        return dict(matches)

    def decode_amr(self, lines: str) -> penman.Graph:
        """
        Decode string using the AMR codec.
        """
        return self.amr_codec.decode(lines)

    @classmethod
    def process_var_line(cls, line, var_dict):
        """
        Function that processes line with a variable in it. Returns the string without
        variables and the dictionary with var-name + var - value
        Only works if AMR is shown as multiple lines and input correctly!

        Based on Noord and Bos (2017) pre-processing scripts.
        """

        curr_var_name = False
        curr_var_value = False
        var_value = ''
        var_name = ''

        for idx, ch in enumerate(line):
            if ch == '/':  # we start adding the variable value
                curr_var_value = True
                curr_var_name = False
                var_value = ''
                continue

            if ch == '(':  # we start adding the variable name
                curr_var_name = True
                curr_var_value = False
                if var_value and var_name:  # we already found a name-value pair, add it now
                    var_dict[var_name.strip()] = var_value.strip().replace(')', '').replace(' :name', '').replace(
                        ' :dayperiod', '').replace(' :mod', '')
                var_name = ''
                continue

            if curr_var_name:  # add to variable name
                var_name += ch
            if curr_var_value:  # add to variable value
                var_value += ch

        var_dict[var_name.strip()] = var_value.strip().replace(')', '')
        deleted_var_string = re.sub(r'\((.*?/)', '(', line).replace('( ', '(')  # delete variables from line

        return deleted_var_string, var_dict

    @classmethod
    def delete_wiki(cls, lines: Lines) -> Lines:
        """
        Delete wiki links from AMR

        Based on Noord and Bos (2017) pre-processing scripts.
        """

        no_wiki = []

        for line in lines:
            n_line = re.sub(r':wiki "(.*?)"', '', line, 1)
            n_line = re.sub(':wiki -', '', n_line)
            no_wiki.append(
                (len(n_line) - len(n_line.lstrip())) * ' ' + ' '.join(
                    n_line.split()))  # convert double whitespace but keep leading whitespace

        return no_wiki

    @classmethod
    def delete_amr_variables(cls, lines: Lines) -> Lines:
        """
        Function that deletes variable names from AMR

        Based on Noord and Bos (2017) pre-processing scripts.
        """

        var_dict = dict()
        variables_removed = []

        for line in lines:
            if '/' in line:  # variable here
                deleted_var_string, var_dict = cls.process_var_line(line, var_dict)  # process line and save variables
                variables_removed.append(deleted_var_string)  # save string with variables deleted

            else:  # (probable) reference to variable here!
                split_line = line.split()
                ref_var = split_line[1].replace(')', '')  # get var name

                if ref_var in var_dict:
                    ref_value = var_dict[
                        ref_var]  # value to replace the variable name with
                    split_line[1] = split_line[1].replace(
                        ref_var, '(' + ref_value.strip() + ')')  # do the replacing and add brackets for alignment
                    n_line = (len(line) - len(
                        line.lstrip())) * ' ' + " ".join(split_line)
                    variables_removed.append(n_line)
                else:
                    variables_removed.append(
                        line)  # no reference found, add line without editing (usually there are numbers in this line)
        return variables_removed

    @classmethod
    def single_line_convert(cls, lines: Lines) -> str:
        """
        Convert AMR to a single line

        Based on Noord and Bos (2017) pre-processing scripts.
        """

        # TODO: Replace with regex replacement of '\n's and multiple spaces with ' '
        amr = []
        for line in lines:
            amr.append(line.strip())
        return " ".join(amr)

    @classmethod
    def linearize_amr(cls, amr_lines: str) -> str:
        """
        Returns pre-processed linearized amr.
        """
        amr_lines = amr_lines.split('\n')
        amr_lines = cls.delete_wiki(amr_lines)
        amr_lines = cls.delete_amr_variables(amr_lines)
        one_line_amr = cls.single_line_convert(amr_lines)

        return one_line_amr

    def parse_instance(self, *,
                       snt: str = None,
                       metadata: str = None,
                       amr: str = None) -> Instance:
        """
        Parse AllenNLP Instances.
        """
        fields: Dict[str, Field] = {}

        if metadata is None:
            # Sometimes metadata may not given: e.g. only raw sentence is given.
            # If no metadata available, automatically generate id by hash.
            # Having id may be useful for debugging.
            snt_hash = zlib.adler32(snt.encode('utf-8'))
            metadata = {
                'id': f'anonymous_{snt_hash:08x}',
                'snt': snt
            }
        else:
            metadata = self.decode_metadata(metadata)
            snt = metadata['snt']
        fields['metadata'] = MetadataField(metadata)

        # Preparing sentence as a text field
        snt_tokens = self.snt_tokenizer.tokenize(snt)
        fields['snt'] = TextField(snt_tokens, token_indexers=self.snt_token_indexers)

        # In inference mode sample may be given without gold labels.
        if amr is None:
            return Instance(fields)

        # Prepare linearized AMR as text token field
        linearized = self.linearize_amr(amr)
        linearized_tokens = self.linearized_amr_tokenizer.tokenize(linearized)
        fields['amr_linearized'] = TextField(linearized_tokens,
                                             token_indexers=self.linearized_amr_indexers)

        if self.graph:
            try:
                amr = self.decode_amr(amr)
                fields['amr_graph'] = AMRGraphField(amr, token_indexers=self.linearized_amr_indexers)
            except Exception as e:  # penman.DecodeError
                logging.warning(e)
                amr = self.decode_amr('(e / error)')
                fields['amr_graph'] = AMRGraphField(amr, token_indexers=self.linearized_amr_indexers)

        # Raw AMR may be used to calculate metrics in evaluation phase.
        fields['raw_amr'] = MetadataField(amr)

        return Instance(fields)

    # some typing annotations as references

    @overload
    def text_to_instance(self, block: str) -> Instance:
        """
        Parse Instance of sentence, metadata and AMR given raw block text.
        Intended to use with predictors.

        Arguments:
            block: Text block of AMR-sentence pair in `LDC2017T10` format.
                AMR-sentence pair encoding specification.
        Input example:
            # ::id bolt12_10495_0392.5 ::date 2012-12-12T20:30:09 ::annotator SDL-AMR-09 ::preferred
            # ::snt Here it is a country with the freedom of speech.
            # ::save-date Tue Apr 28, 2015 ::file bolt12_10495_0392_5.txt
            (c / country
                  :ARG1-of (f / free-04
                        :ARG3 (s / speak-01))
                  :location (h / here)
                  :domain (i / it))
        References:
             [AMR-sentence pair encoding specification](https://catalog.ldc.upenn.edu/docs/LDC2017T10/README.txt),
             Section `2.3 Structure and content of individual AMRs`
        """
        ...  # typing annotation only

    @overload
    def text_to_instance(self, *, snt: str) -> Instance:
        """
        Parse Instance of given sentence only.
        Intended to use with predictors.

        Arguments:
            snt: Raw sentence text.
        """
        ...  # typing annotation only

    @overload
    def text_to_instance(self, *, comments: str, contents: str) -> Instance:
        """
        Parse Instance of sentence, metadata and AMR given comments and content
        sections of `LDC2017T10` AMR-sentence pair format.
        Must use for training stage.

        Arguments:
            comments: Comments section, to extract metadata and sentence from.
            contents: Contents section, to extract PENMAN-encoded AMR from.
        References:
             [AMR-sentence pair encoding specification](https://catalog.ldc.upenn.edu/docs/LDC2017T10/README.txt),
             `2.3 Structure and content of individual AMRs`
        """
        ...  # typing annotation only

    def text_to_instance(self, block: str = None, *,
                         snt: str = None,
                         comments: str = None,
                         contents: str = None) -> Instance:
        if block is not None:
            block = block.strip()
            if not block.startswith('#'):
                return self.parse_instance(snt=block)
            # if raw block text is given, treat as a file of one block
            block_io = io.StringIO(block)
            instance, = self._read_io(block_io)  # checks if yields exactly one Instance
            return instance

        if snt is not None:
            return self.parse_instance(snt=snt)

        if comments is not None and contents is not None:
            return self.parse_instance(metadata=comments,
                                       amr=contents)

        raise ValueError

    @classmethod
    def read_blocks(cls, file: Iterator[str]) -> Iterator[Tuple[Lines, Lines]]:
        """
        Read stream of `LDC2017T10` AMR-sentence pair format.
        The function is a generator yielding pairs of comments (metadata)
        and contents (AMRs in PENMAN format).
        """
        comments: Lines = []
        contents: Lines = []
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                # Metadata starts with #
                line = line.lstrip('#')
                # If we have found metadata line then return contents
                # found before (if there are any)
                if contents:
                    yield comments, contents
                    comments = []
                    contents = []
                # Store metadata
                comments.append(line)
            else:
                # Store contents
                contents.append(line)
        # If there are any contents left, return them
        if contents:
            yield comments, contents

    def _read_io(self, file: TextIO,
                 **kwargs):  # TODO add metadata
        """
        Read file and yield parsed instances.
        """
        for comments, contents in self.read_blocks(file):
            comments = os.linesep.join(comments)
            contents = os.linesep.join(contents)
            try:
                yield self.text_to_instance(comments=comments,
                                            contents=contents)
            except Exception as e:
                # Don't yield bad samples.
                logging.warning('bad case')

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        """
        Read given file(s).
        """
        paths = glob(file_path, recursive=True)

        for path in sorted(paths):
            with open(path, 'r', encoding='utf-8') as f:
                yield from self._read_io(f, file_path=path)
