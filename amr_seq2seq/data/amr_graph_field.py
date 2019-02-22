from typing import List, Dict, Tuple

from overrides import overrides

import torch
from allennlp.data.fields import Field, TextField, AdjacencyField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer

import penman


from collections import OrderedDict
from allennlp.data import DataArray, Vocabulary


class AMRGraphField(Field[Dict[str, torch.Tensor]]):

    def __init__(self, amr_graph: penman.Graph, *,
                 token_indexers: Dict[str, TokenIndexer]):
        super().__init__()
        self.amr_graph = amr_graph
        self.token_indexers = token_indexers

        self.vars = self.get_vars()
        self.var_tokens = [self.get_token(var)
                           for var in self.vars]
        self.tokens_field = TextField(self.var_tokens,
                                      token_indexers=self.token_indexers)

        relations: Dict[Tuple[int, int], str] = {}

        for triple in self.get_edges():
            source_id = self.vars[triple.source]
            target_id = self.vars[triple.target]
            # TODO multiple relations between two nodes are dismissed
            relations[source_id, target_id] = triple.relation

        adjacency_indices = list(relations.keys())
        adjacency_labels = list(relations.values())

        # self.top = self.get_top()
        # self.top_field = IndexField(self.vars[self.top],
        #                             sequence_field=self.tokens_field)

        self.adjacency_field = AdjacencyField(adjacency_indices,
                                              sequence_field=self.tokens_field,
                                              labels=adjacency_labels,
                                              label_namespace='amr_edges')

    def get_top(self) -> str:
        return self.amr_graph.top

    def get_vars(self) -> OrderedDict:
        variables = OrderedDict()
        for idx, rel in enumerate(self.amr_graph.triples(relation='instance')):
            variables[rel.source] = idx
        return variables

    def get_token(self, var: str) -> Token:
        instances: List[penman.Triple] = self.amr_graph.triples(var, relation='instance')
        # assert len(instances) == 1  # TODO
        if len(instances) != 1:
            print()
        token_str = instances[0].target
        return Token(token_str)

    def get_edges(self,
                  source: str = None,
                  relation: str = None,
                  target: str = None) -> List[penman.Triple]:
        return self.amr_graph.edges(source, relation, target)

    @overrides
    def as_tensor(self,
                  padding_lengths: Dict[str, int]) -> Dict[str, DataArray]:
        return {
            'nodes': self.tokens_field.as_tensor(padding_lengths),
            'edges': self.adjacency_field.as_tensor(padding_lengths),
            # 'top': self.top_field.as_tensor(padding_lengths),
        }

    @overrides
    def batch_tensors(self, tensor_list: List[Dict[str, DataArray]]) -> Dict[str, DataArray]:
        nodes = [tensors['nodes'] for tensors in tensor_list]
        edges = [tensors['edges'] for tensors in tensor_list]
        # top = [tensors['top'] for tensors in tensor_list]
        return {
            'nodes': self.tokens_field.batch_tensors(nodes),
            'edges': self.adjacency_field.batch_tensors(edges),
            # 'top': self.top_field.batch_tensors(top)
        }

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return self.tokens_field.get_padding_lengths()

    @overrides
    def index(self, vocab: Vocabulary):
        self.tokens_field.index(vocab)
        self.adjacency_field.index(vocab)

    @overrides
    def empty_field(self) -> Field:
        raise NotImplementedError

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        self.tokens_field.count_vocab_items(counter)
        self.adjacency_field.count_vocab_items(counter)
        # self.top_field.count_vocab_items(counter)
