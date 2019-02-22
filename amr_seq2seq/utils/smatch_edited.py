#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
This script computes smatch score between two AMRs.
For detailed description of smatch, see http://www.isi.edu/natural-language/amr/smatch-13.pdf

"""

from __future__ import print_function
from __future__ import division

try:
    from . import amr
except:
    import amr

import os
import logging
import random
import sys
import argparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)


def build_arg_parser():
    """
    Build an argument parser using argparse. Use it when python version is 2.7 or later.

    """
    parser = argparse.ArgumentParser(description="Smatch calculator -- arguments")
    parser.add_argument('-f', nargs=2, required=True, type=str,
                        help='Two files containing AMR pairs. AMRs in each file are separated by a single blank line')
    parser.add_argument('-r', type=int, default=4, help='Restart number (Default:4)')
    parser.add_argument('--significant', type=int, default=4, help='significant digits to output (default: 2)')
    parser.add_argument('-v', action='store_true', help='Verbose output (Default:false)')
    parser.add_argument('--vv', action='store_true', help='Very Verbose output (Default:false)')
    parser.add_argument('--ms', action='store_true', default=False,
                        help='Output multiple scores (one AMR pair a score)'
                             'instead of a single document-level smatch score (Default: false)')
    parser.add_argument('--pr', action='store_true', default=False,
                        help="Output precision and recall as well as the f-score. Default: false")
    parser.add_argument('--one_line', default='prod', choices=['no', 'prod', 'gold', 'both'], type=str,
                        help="If the input is in one-line format (default prod)")
    parser.add_argument('--justinstance', action='store_true', default=False,
                        help="just pay attention to matching instances")
    parser.add_argument('--justattribute', action='store_true', default=False,
                        help="just pay attention to matching attributes")
    parser.add_argument('--justrelation', action='store_true', default=False,
                        help="just pay attention to matching relations")

    return parser


class SmatchScript:

    def __init__(self,
                 r=4,
                 significant=4,
                 v=False,
                 vv=False,
                 ms=False,
                 pr=False,
                 justinstance=False,
                 justattribute=False,
                 justrelation=False,
                 **kwargs):
        # logging.critical(r,significant,v,vv,ms,pr,justinstance,justattribute,justrelation,kwargs)
        # total number of iteration in smatch computation
        self.iteration_num = 5

        # verbose output switch.
        # Default false (no verbose output)
        self.verbose = False
        self.veryVerbose = False

        # single score output switch.
        # Default true (compute a single score for all AMRs in two files)
        self.single_score = True

        # precision and recall output switch.
        # Default false (do not output precision and recall, just output F score)
        self.pr_flag = False

        # dictionary to save pre-computed node mapping and its resulting triple match count
        # key: tuples of node mapping
        # value: the matching triple count
        self.match_triple_dict = {}

        self.verbose = v
        self.veryVerbose = vv
        self.pr_flag = pr
        self.arguments_ms = ms

        self.doinstance = True
        self.doattribute = True
        self.dorelation = True

        self.justinstance = justinstance
        self.justattribute = justattribute
        self.justrelation = justrelation

        self.significant = significant

        # matching triple number
        self.total_match_num = 0
        # triple number in test file
        self.total_test_num = 0
        # triple number in gold file
        self.total_gold_num = 0
        # sentence number
        self.sent_num = 1
        # significant digits to print out
        self.floatdisplay = "%%.%df" % self.significant
        # Read amr pairs from two files

        self.iteration_num = r + 1
        if self.arguments_ms:
            self.single_score = False

        if self.justinstance:
            self.doattribute = False
            self.dorelation = False

        if self.justattribute:
            self.doinstance = False
            self.dorelation = False

        if self.justrelation:
            self.doinstance = False
            self.doattribute = False

    @staticmethod
    def get_amr_line(input_f):
        """
        Read the file containing AMRs. AMRs are separated by a blank line.
        Each call of get_amr_line() returns the next available AMR (in one-line form).
        Note: this function does not verify if the AMR is valid"""

        all_amrs = []
        cur_amr = []
        has_content = False

        for line in open(input_f, 'r'):
            line = line.strip()
            if line == "":
                if not has_content:
                    # empty lines before current AMR
                    continue
                else:
                    # end of current AMR
                    all_amrs.append("".join(cur_amr))
                    cur_amr = []
                    has_content = False
                    continue
            if line.strip().startswith("#"):
                # ignore the comment line (starting with "#") in the AMR file
                continue
            else:
                has_content = True
                cur_amr.append(line.strip())

        if cur_amr != []:
            all_amrs.append("".join(cur_amr))

        return all_amrs

    def get_best_match(self, instance1, attribute1, relation1,
                       instance2, attribute2, relation2,
                       prefix1, prefix2, doinstance=True, doattribute=True, dorelation=True):
        """
        Get the highest triple match number between two sets of triples via hill-climbing.
        Arguments:
            instance1: instance triples of AMR 1 ("instance", node name, node value)
            attribute1: attribute triples of AMR 1 (attribute name, node name, attribute value)
            relation1: relation triples of AMR 1 (relation name, node 1 name, node 2 name)
            instance2: instance triples of AMR 2 ("instance", node name, node value)
            attribute2: attribute triples of AMR 2 (attribute name, node name, attribute value)
            relation2: relation triples of AMR 2 (relation name, node 1 name, node 2 name)
            prefix1: prefix label for AMR 1
            prefix2: prefix label for AMR 2
        Returns:
            best_match: the node mapping that results in the highest triple matching number
            best_match_num: the highest triple matching number

        """
        # Compute candidate pool - all possible node match candidates.
        # In the hill-climbing, we only consider candidate in this pool to save computing time.
        # weight_dict is a dictionary that maps a pair of node
        (candidate_mappings, weight_dict) = self.compute_pool(instance1, attribute1, relation1,
                                                              instance2, attribute2, relation2,
                                                              prefix1, prefix2, doinstance=doinstance,
                                                              doattribute=doattribute,
                                                              dorelation=dorelation)
        if self.veryVerbose:
            logger.info("Candidate mappings:")
            logger.info(candidate_mappings)
            logger.info("Weight dictionary")
            logger.info(weight_dict)

        best_match_num = 0
        # initialize best match mapping
        # the ith entry is the node index in AMR 2 which maps to the ith node in AMR 1
        best_mapping = [-1] * len(instance1)
        for i in range(self.iteration_num):
            if self.veryVerbose:
                logger.info("Iteration", i)
            if i == 0:
                # smart initialization used for the first round
                cur_mapping = self.smart_init_mapping(candidate_mappings, instance1, instance2)
            else:
                # random initialization for the other round
                cur_mapping = self.random_init_mapping(candidate_mappings)
            # compute current triple match number
            match_num = self.compute_match(cur_mapping, weight_dict)
            if self.veryVerbose:
                logger.info("Node mapping at start", cur_mapping)
                logger.info("Triple match number at start:", match_num)
            while True:
                # get best gain
                (gain, new_mapping) = self.get_best_gain(cur_mapping, candidate_mappings, weight_dict,
                                                         len(instance2), match_num)
                if self.veryVerbose:
                    logger.info("Gain after the hill-climbing", gain)
                # hill-climbing until there will be no gain for new node mapping
                if gain <= 0:
                    break
                # otherwise update match_num and mapping
                match_num += gain
                cur_mapping = new_mapping[:]
                if self.veryVerbose:
                    logger.info("Update triple match number to:", match_num)
                    logger.info("Current mapping:", cur_mapping)
            if match_num > best_match_num:
                best_mapping = cur_mapping[:]
                best_match_num = match_num
        return best_mapping, best_match_num

    @staticmethod
    def normalize(item):
        """
        lowercase and remove quote signifiers from items that are about to be compared
        """
        return item.lower().rstrip('_')

    def compute_pool(self, instance1, attribute1, relation1,
                     instance2, attribute2, relation2,
                     prefix1, prefix2, doinstance=True, doattribute=True, dorelation=True):
        """
        compute all possible node mapping candidates and their weights (the triple matching number gain resulting from
        mapping one node in AMR 1 to another node in AMR2)

        Arguments:
            instance1: instance triples of AMR 1
            attribute1: attribute triples of AMR 1 (attribute name, node name, attribute value)
            relation1: relation triples of AMR 1 (relation name, node 1 name, node 2 name)
            instance2: instance triples of AMR 2
            attribute2: attribute triples of AMR 2 (attribute name, node name, attribute value)
            relation2: relation triples of AMR 2 (relation name, node 1 name, node 2 name
            prefix1: prefix label for AMR 1
            prefix2: prefix label for AMR 2
        Returns:
          candidate_mapping: a list of candidate nodes.
                           The ith element contains the node indices (in AMR 2) the ith node (in AMR 1) can map to.
                           (resulting in non-zero triple match)
          weight_dict: a dictionary which contains the matching triple number for every pair of node mapping. The key
                       is a node pair. The value is another dictionary. key {-1} is triple match resulting from this node
                       pair alone (instance triples and attribute triples), and other keys are node pairs that can result
                       in relation triple match together with the first node pair.


        """
        candidate_mapping = []
        weight_dict = {}
        for i in range(0, len(instance1)):  # TODO
            # each candidate mapping is a set of node indices
            candidate_mapping.append(set())
            if doinstance:
                for j in range(0, len(instance2)):  # TODO
                    # if both triples are instance triples and have the same value
                    # get nodes that can possibly match
                    if self.normalize(instance1[i][0]) == self.normalize(instance2[j][0]) and \
                            self.normalize(instance1[i][2]) == self.normalize(instance2[j][2]):
                        # print 'match:\n'
                        # print instance1[i][0], instance1[j][0]
                        # print instance1[i][2], instance1[i][2],'\n'
                        # get node index by stripping the prefix
                        node1_index = int(instance1[i][1][len(prefix1):])
                        node2_index = int(instance2[j][1][len(prefix2):])
                        candidate_mapping[node1_index].add(node2_index)
                        node_pair = (node1_index, node2_index)
                        # use -1 as key in weight_dict for instance triples and attribute triples
                        if node_pair in weight_dict:
                            weight_dict[node_pair][-1] += 1
                        else:
                            weight_dict[node_pair] = {}
                            weight_dict[node_pair][-1] = 1
        if doattribute:
            for i in range(0, len(attribute1)):
                for j in range(0, len(attribute2)):
                    # if both attribute relation triple have the same relation name and value
                    if self.normalize(attribute1[i][0]) == self.normalize(attribute2[j][0]) \
                            and self.normalize(attribute1[i][2]) == self.normalize(attribute2[j][2]):
                        node1_index = int(attribute1[i][1][len(prefix1):])
                        node2_index = int(attribute2[j][1][len(prefix2):])
                        candidate_mapping[node1_index].add(node2_index)
                        node_pair = (node1_index, node2_index)
                        # use -1 as key in weight_dict for instance triples and attribute triples
                        if node_pair in weight_dict:
                            weight_dict[node_pair][-1] += 1
                        else:
                            weight_dict[node_pair] = {}
                            weight_dict[node_pair][-1] = 1

        if dorelation:
            for i in range(0, len(relation1)):
                for j in range(0, len(relation2)):
                    # if both relation share the same name
                    if self.normalize(relation1[i][0]) == self.normalize(relation2[j][0]):
                        node1_index_amr1 = int(relation1[i][1][len(prefix1):])
                        node1_index_amr2 = int(relation2[j][1][len(prefix2):])
                        node2_index_amr1 = int(relation1[i][2][len(prefix1):])
                        node2_index_amr2 = int(relation2[j][2][len(prefix2):])
                        # add mapping between two nodes
                        candidate_mapping[node1_index_amr1].add(node1_index_amr2)
                        candidate_mapping[node2_index_amr1].add(node2_index_amr2)
                        node_pair1 = (node1_index_amr1, node1_index_amr2)
                        node_pair2 = (node2_index_amr1, node2_index_amr2)
                        if node_pair2 != node_pair1:
                            # update weight_dict weight. Note that we need to update both entries for future search
                            # i.e weight_dict[node_pair1][node_pair2]
                            #     weight_dict[node_pair2][node_pair1]
                            if node1_index_amr1 > node2_index_amr1:
                                # swap node_pair1 and node_pair2
                                node_pair1 = (node2_index_amr1, node2_index_amr2)
                                node_pair2 = (node1_index_amr1, node1_index_amr2)
                            if node_pair1 in weight_dict:
                                if node_pair2 in weight_dict[node_pair1]:
                                    weight_dict[node_pair1][node_pair2] += 1
                                else:
                                    weight_dict[node_pair1][node_pair2] = 1
                            else:
                                weight_dict[node_pair1] = {}
                                weight_dict[node_pair1][-1] = 0
                                weight_dict[node_pair1][node_pair2] = 1
                            if node_pair2 in weight_dict:
                                if node_pair1 in weight_dict[node_pair2]:
                                    weight_dict[node_pair2][node_pair1] += 1
                                else:
                                    weight_dict[node_pair2][node_pair1] = 1
                            else:
                                weight_dict[node_pair2] = {}
                                weight_dict[node_pair2][-1] = 0
                                weight_dict[node_pair2][node_pair1] = 1
                        else:
                            # two node pairs are the same. So we only update weight_dict once.
                            # this generally should not happen.
                            if node_pair1 in weight_dict:
                                weight_dict[node_pair1][-1] += 1
                            else:
                                weight_dict[node_pair1] = {}
                                weight_dict[node_pair1][-1] = 1
        # print 'len weight dict: {0}'.format(len(weight_dict))
        # print weight_dict,'\n\n'
        # print 'len candidate mapping: {0}'.format(len(candidate_mapping))
        # print candidate_mapping ,'\n'

        return candidate_mapping, weight_dict

    @staticmethod
    def smart_init_mapping(candidate_mapping, instance1, instance2):
        """
        Initialize mapping based on the concept mapping (smart initialization)
        Arguments:
            candidate_mapping: candidate node match list
            instance1: instance triples of AMR 1
            instance2: instance triples of AMR 2
        Returns:
            initialized node mapping between two AMRs

        """
        random.seed()
        matched_dict = {}
        result = []
        # list to store node indices that have no concept match
        no_word_match = []
        for i, candidates in enumerate(candidate_mapping):
            if len(candidates) == 0:
                # no possible mapping
                result.append(-1)
                continue
            # node value in instance triples of AMR 1
            value1 = instance1[i][2]
            for node_index in candidates:
                value2 = instance2[node_index][2]
                # find the first instance triple match in the candidates
                # instance triple match is having the same concept value
                if value1 == value2:
                    if node_index not in matched_dict:
                        result.append(node_index)
                        matched_dict[node_index] = 1
                        break
            if len(result) == i:
                no_word_match.append(i)
                result.append(-1)
        # if no concept match, generate a random mapping
        for i in no_word_match:
            candidates = list(candidate_mapping[i])
            while len(candidates) > 0:
                # get a random node index from candidates
                rid = random.randint(0, len(candidates) - 1)
                if candidates[rid] in matched_dict:
                    candidates.pop(rid)
                else:
                    matched_dict[candidates[rid]] = 1
                    result[i] = candidates[rid]
                    break
        return result

    @staticmethod
    def random_init_mapping(candidate_mapping):
        """
        Generate a random node mapping.
        Args:
            candidate_mapping: candidate_mapping: candidate node match list
        Returns:
            randomly-generated node mapping between two AMRs

        """
        # if needed, a fixed seed could be passed here to generate same random (to help debugging)
        random.seed()
        matched_dict = {}
        result = []
        for c in candidate_mapping:
            candidates = list(c)
            if len(candidates) == 0:
                # -1 indicates no possible mapping
                result.append(-1)
                continue
            found = False
            while len(candidates) > 0:
                # randomly generate an index in [0, length of candidates)
                rid = random.randint(0, len(candidates) - 1)
                # check if it has already been matched
                if candidates[rid] in matched_dict:
                    candidates.pop(rid)
                else:
                    matched_dict[candidates[rid]] = 1
                    result.append(candidates[rid])
                    found = True
                    break
            if not found:
                result.append(-1)
        return result

    def compute_match(self, mapping, weight_dict):
        """
        Given a node mapping, compute match number based on weight_dict.
        Args:
        mappings: a list of node index in AMR 2. The ith element (value j) means node
                  i in AMR 1 maps to node j in AMR 2.
        Returns:
        matching triple number
        Complexity: O(m*n) , m is the node number of AMR 1, n is the node number of AMR 2

        """
        # If this mapping has been investigated before, retrieve the value instead of re-computing.
        if self.veryVerbose:
            logger.info("Computing match for mapping")
            logger.info(mapping)
        if tuple(mapping) in self.match_triple_dict:
            if self.veryVerbose:
                logger.info("saved value", self.match_triple_dict[tuple(mapping)])
            return self.match_triple_dict[tuple(mapping)]
        match_num = 0
        # i is node index in AMR 1, m is node index in AMR 2
        for i, m in enumerate(mapping):
            if m == -1:
                # no node maps to this node
                continue
            # node i in AMR 1 maps to node m in AMR 2
            current_node_pair = (i, m)
            if current_node_pair not in weight_dict:
                continue
            if self.veryVerbose:
                logger.info("node_pair", current_node_pair)
            for key in weight_dict[current_node_pair]:
                if key == -1:
                    # matching triple resulting from instance/attribute triples
                    match_num += weight_dict[current_node_pair][key]
                    if self.veryVerbose:
                        logger.info("instance/attribute match", weight_dict[current_node_pair][key])
                # only consider node index larger than i to avoid duplicates
                # as we store both weight_dict[node_pair1][node_pair2] and
                #     weight_dict[node_pair2][node_pair1] for a relation
                elif key[0] < i:
                    continue
                elif mapping[key[0]] == key[1]:
                    match_num += weight_dict[current_node_pair][key]
                    if self.veryVerbose:
                        logger.info("relation match with", key, weight_dict[current_node_pair][key])
        if self.veryVerbose:
            logger.info("match computing complete, result:", match_num)
        # update match_triple_dict
        self.match_triple_dict[tuple(mapping)] = match_num
        return match_num

    def move_gain(self, mapping, node_id, old_id, new_id, weight_dict, match_num):
        """
        Compute the triple match number gain from the move operation
        Arguments:
            mapping: current node mapping
            node_id: remapped node in AMR 1
            old_id: original node id in AMR 2 to which node_id is mapped
            new_id: new node in to which node_id is mapped
            weight_dict: weight dictionary
            match_num: the original triple matching number
        Returns:
            the triple match gain number (might be negative)

        """
        # new node mapping after moving
        new_mapping = (node_id, new_id)
        # node mapping before moving
        old_mapping = (node_id, old_id)
        # new nodes mapping list (all node pairs)
        new_mapping_list = mapping[:]
        new_mapping_list[node_id] = new_id
        # if this mapping is already been investigated, use saved one to avoid duplicate computing
        if tuple(new_mapping_list) in self.match_triple_dict:
            return self.match_triple_dict[tuple(new_mapping_list)] - match_num
        gain = 0
        # add the triple match incurred by new_mapping to gain
        if new_mapping in weight_dict:
            for key in weight_dict[new_mapping]:
                if key == -1:
                    # instance/attribute triple match
                    gain += weight_dict[new_mapping][-1]
                elif new_mapping_list[key[0]] == key[1]:
                    # relation gain incurred by new_mapping and another node pair in new_mapping_list
                    gain += weight_dict[new_mapping][key]
        # deduct the triple match incurred by old_mapping from gain
        if old_mapping in weight_dict:
            for k in weight_dict[old_mapping]:
                if k == -1:
                    gain -= weight_dict[old_mapping][-1]
                elif mapping[k[0]] == k[1]:
                    gain -= weight_dict[old_mapping][k]
        # update match number dictionary
        self.match_triple_dict[tuple(new_mapping_list)] = match_num + gain
        return gain

    def swap_gain(self, mapping, node_id1, mapping_id1, node_id2, mapping_id2, weight_dict, match_num):
        """
        Compute the triple match number gain from the swapping
        Arguments:
        mapping: current node mapping list
        node_id1: node 1 index in AMR 1
        mapping_id1: the node index in AMR 2 node 1 maps to (in the current mapping)
        node_id2: node 2 index in AMR 1
        mapping_id2: the node index in AMR 2 node 2 maps to (in the current mapping)
        weight_dict: weight dictionary
        match_num: the original matching triple number
        Returns:
        the gain number (might be negative)

        """
        new_mapping_list = mapping[:]
        # Before swapping, node_id1 maps to mapping_id1, and node_id2 maps to mapping_id2
        # After swapping, node_id1 maps to mapping_id2 and node_id2 maps to mapping_id1
        new_mapping_list[node_id1] = mapping_id2
        new_mapping_list[node_id2] = mapping_id1
        if tuple(new_mapping_list) in self.match_triple_dict:
            return self.match_triple_dict[tuple(new_mapping_list)] - match_num
        gain = 0
        new_mapping1 = (node_id1, mapping_id2)
        new_mapping2 = (node_id2, mapping_id1)
        old_mapping1 = (node_id1, mapping_id1)
        old_mapping2 = (node_id2, mapping_id2)
        if node_id1 > node_id2:
            new_mapping2 = (node_id1, mapping_id2)
            new_mapping1 = (node_id2, mapping_id1)
            old_mapping1 = (node_id2, mapping_id2)
            old_mapping2 = (node_id1, mapping_id1)
        if new_mapping1 in weight_dict:
            for key in weight_dict[new_mapping1]:
                if key == -1:
                    gain += weight_dict[new_mapping1][-1]
                elif new_mapping_list[key[0]] == key[1]:
                    gain += weight_dict[new_mapping1][key]
        if new_mapping2 in weight_dict:
            for key in weight_dict[new_mapping2]:
                if key == -1:
                    gain += weight_dict[new_mapping2][-1]
                # to avoid duplicate
                elif key[0] == node_id1:
                    continue
                elif new_mapping_list[key[0]] == key[1]:
                    gain += weight_dict[new_mapping2][key]
        if old_mapping1 in weight_dict:
            for key in weight_dict[old_mapping1]:
                if key == -1:
                    gain -= weight_dict[old_mapping1][-1]
                elif mapping[key[0]] == key[1]:
                    gain -= weight_dict[old_mapping1][key]
        if old_mapping2 in weight_dict:
            for key in weight_dict[old_mapping2]:
                if key == -1:
                    gain -= weight_dict[old_mapping2][-1]
                # to avoid duplicate
                elif key[0] == node_id1:
                    continue
                elif mapping[key[0]] == key[1]:
                    gain -= weight_dict[old_mapping2][key]
        self.match_triple_dict[tuple(new_mapping_list)] = match_num + gain
        return gain

    def get_best_gain(self, mapping, candidate_mappings, weight_dict, instance_len, cur_match_num):
        """
        Hill-climbing method to return the best gain swap/move can get
        Arguments:
        mapping: current node mapping
        candidate_mappings: the candidates mapping list
        weight_dict: the weight dictionary
        instance_len: the number of the nodes in AMR 2
        cur_match_num: current triple match number
        Returns:
        the best gain we can get via swap/move operation

        """
        largest_gain = 0
        # True: using swap; False: using move
        use_swap = True
        # the node to be moved/swapped
        node1 = None
        # store the other node affected. In swap, this other node is the node swapping with node1. In move, this other
        # node is the node node1 will move to.
        node2 = None
        # unmatched nodes in AMR 2
        unmatched = set(range(instance_len))
        # exclude nodes in current mapping
        # get unmatched nodes
        for nid in mapping:
            if nid in unmatched:
                unmatched.remove(nid)
        for i, nid in enumerate(mapping):
            # current node i in AMR 1 maps to node nid in AMR 2
            for nm in unmatched:
                if nm in candidate_mappings[i]:
                    # remap i to another unmatched node (move)
                    # (i, m) -> (i, nm)
                    if self.veryVerbose:
                        logger.info("Remap node", i, "from ", nid, "to", nm)
                    mv_gain = self.move_gain(mapping, i, nid, nm, weight_dict, cur_match_num)
                    if self.veryVerbose:
                        logger.info("Move gain:", mv_gain)
                        new_mapping = mapping[:]
                        new_mapping[i] = nm
                        new_match_num = self.compute_match(new_mapping, weight_dict)
                        if new_match_num != cur_match_num + mv_gain:
                            logger.error(mapping, new_mapping)
                            logger.error("Inconsistency in computing: move gain", cur_match_num, mv_gain, new_match_num)
                    if mv_gain > largest_gain:
                        largest_gain = mv_gain
                        node1 = i
                        node2 = nm
                        use_swap = False
        # compute swap gain
        for i, m in enumerate(mapping):
            for j in range(i + 1, len(mapping)):
                m2 = mapping[j]
                # swap operation (i, m) (j, m2) -> (i, m2) (j, m)
                # j starts from i+1, to avoid duplicate swap
                if self.veryVerbose:
                    logger.info("Swap node", i, "and", j)
                    logger.info("Before swapping:", i, "-", m, ",", j, "-", m2)
                    logger.info(mapping)
                    logger.info("After swapping:", i, "-", m2, ",", j, "-", m)
                sw_gain = self.swap_gain(mapping, i, m, j, m2, weight_dict, cur_match_num)
                if self.veryVerbose:
                    logger.info("Swap gain:", sw_gain)
                    new_mapping = mapping[:]
                    new_mapping[i] = m2
                    new_mapping[j] = m
                    logger.info(new_mapping)
                    new_match_num = self.compute_match(new_mapping, weight_dict)
                    if new_match_num != cur_match_num + sw_gain:
                        logger.error(mapping, new_mapping)
                        logger.error("Inconsistency in computing: swap gain", cur_match_num, sw_gain, new_match_num)
                if sw_gain > largest_gain:
                    largest_gain = sw_gain
                    node1 = i
                    node2 = j
                    use_swap = True
        # generate a new mapping based on swap/move
        cur_mapping = mapping[:]
        if node1 is not None:
            if use_swap:
                if self.veryVerbose:
                    logger.info("Use swap gain")
                temp = cur_mapping[node1]
                cur_mapping[node1] = cur_mapping[node2]
                cur_mapping[node2] = temp
            else:
                if self.veryVerbose:
                    logger.info("Use move gain")
                cur_mapping[node1] = node2
        else:
            if self.veryVerbose:
                logger.info("no move/swap gain found")
        if self.veryVerbose:
            logger.info("Original mapping", mapping)
            logger.info("Current mapping", cur_mapping)
        return largest_gain, cur_mapping

    @staticmethod
    def print_alignment(mapping, instance1, instance2):
        """
        print the alignment based on a node mapping
        Args:
            mapping: current node mapping list
            instance1: nodes of AMR 1
            instance2: nodes of AMR 2

        """
        result = []
        for i, m in enumerate(mapping):
            if m == -1:
                result.append(instance1[i][1] + "(" + instance1[i][2] + ")" + "-Null")
            else:
                result.append(instance1[i][1] + "(" + instance1[i][2] + ")" + "-"
                              + instance2[m][1] + "(" + instance2[m][2] + ")")
        return " ".join(result)

    def compute_f(self, match_num, test_num, gold_num):
        """
        Compute the f-score based on the matching triple number,
                                     triple number of AMR set 1,
                                     triple number of AMR set 2
        Args:
            match_num: matching triple number
            test_num:  triple number of AMR 1 (test file)
            gold_num:  triple number of AMR 2 (gold file)
        Returns:
            precision: match_num/test_num
            recall: match_num/gold_num
            f_score: 2*precision*recall/(precision+recall)
        """
        if test_num == 0 or gold_num == 0:
            return 0.00, 0.00, 0.00
        precision = float(match_num) / float(test_num)
        recall = float(match_num) / float(gold_num)
        if (precision + recall) != 0:
            f_score = 2 * precision * recall / (precision + recall)
            if self.veryVerbose:
                logger.info("F-score:", f_score)
            return precision, recall, f_score
        else:
            if self.veryVerbose:
                logger.info("F-score:", "0.0")
            return precision, recall, 0.00

    def process_instance(self, cur_amr1, cur_amr2):

        # make sure one_line format is given
        cur_amr1 = cur_amr1.replace("\n", "")
        cur_amr2 = cur_amr2.replace("\n", "")

        amr1 = amr.AMR.parse_AMR_line(cur_amr1)
        amr2 = amr.AMR.parse_AMR_line(cur_amr2)

        prefix1 = "a"
        prefix2 = "b"
        # Rename node to "a1", "a2", .etc

        amr1.rename_node(prefix1)
        # Renaming node to "b1", "b2", .etc
        amr2.rename_node(prefix2)
        (instance1, attributes1, relation1) = amr1.get_triples()
        (instance2, attributes2, relation2) = amr2.get_triples()

        if self.verbose:
            # print parse results of two AMRs
            logger.info("AMR pair", self.sent_num)
            logger.info("============================================")
            logger.info("AMR 1 (one-line):", cur_amr1)
            logger.info("AMR 2 (one-line):", cur_amr2)
            logger.info("Instance triples of AMR 1:", len(instance1))
            logger.info(instance1)
            logger.info("Attribute triples of AMR 1:", len(attributes1))
            logger.info(attributes1)
            logger.info("Relation triples of AMR 1:", len(relation1))
            logger.info(relation1)
            logger.info("Instance triples of AMR 2:", len(instance2))
            logger.info(instance2)
            logger.info("Attribute triples of AMR 2:", len(attributes2))
            logger.info(attributes2)
            logger.info("Relation triples of AMR 2:", len(relation2))
            logger.info(relation2)
        (best_mapping, best_match_num) = self.get_best_match(instance1, attributes1, relation1,
                                                             instance2, attributes2, relation2,
                                                             prefix1, prefix2, doinstance=self.doinstance,
                                                             doattribute=self.doattribute, dorelation=self.dorelation)
        if self.verbose:
            logger.info("best match number", best_match_num)
            logger.info("best node mapping", best_mapping)
            logger.info("Best node mapping alignment:", self.print_alignment(best_mapping, instance1, instance2))
        if self.justinstance:
            test_triple_num = len(instance1)
            gold_triple_num = len(instance2)
        elif self.justattribute:
            test_triple_num = len(attributes1)
            gold_triple_num = len(attributes2)
        elif self.justrelation:
            test_triple_num = len(relation1)
            gold_triple_num = len(relation2)
        else:
            test_triple_num = len(instance1) + len(attributes1) + len(relation1)
            gold_triple_num = len(instance2) + len(attributes2) + len(relation2)
        if not self.single_score:
            # if each AMR pair should have a score, compute and output it here
            (precision, recall, best_f_score) = self.compute_f(best_match_num,
                                                               test_triple_num,
                                                               gold_triple_num)
            # print "Sentence", sent_num
            if self.pr_flag:
                print("Precision: " + self.floatdisplay % precision)
                print("Recall: " + self.floatdisplay % recall)
            print("F-score: " + self.floatdisplay % best_f_score)
        self.total_match_num += best_match_num
        self.total_test_num += test_triple_num
        self.total_gold_num += gold_triple_num
        # clear the matching triple dictionary for the next AMR pair
        self.match_triple_dict.clear()
        self.sent_num += 1

    def report(self):

        if self.verbose:
            logger.info("Total match number, total triple number in AMR 1, and total triple number in AMR 2:")
            logger.info(self.total_match_num, self.total_test_num, self.total_gold_num)
            logger.info("---------------------------------------------------------------------------------")
        # output document-level smatch score (a single f-score for all AMR pairs in two files)

        # print('Total test and gold num: {0} and {1}'.format(self.total_test_num, self.total_test_num))

        if self.single_score:
            (precision, recall, best_f_score) = self.compute_f(self.total_match_num, self.total_test_num,
                                                               self.total_gold_num)

            print(self.total_match_num, self.total_test_num, self.total_gold_num)
            if self.pr_flag:
                print("Precision: " + self.floatdisplay % precision)
                print("Recall: " + self.floatdisplay % recall)
            # print('Total AMRs: {0}'.format(len(gold_amrs)))
            print("Document F-score: " + self.floatdisplay % best_f_score)

    def get_metrics(self):
        precision, recall, best_f_score = self.compute_f(self.total_match_num,
                                                         self.total_test_num,
                                                         self.total_gold_num)

        return {
            # "total_match_num": self.total_match_num,
            # "total_test_num": self.total_test_num,
            # "total_gold_num": self.total_gold_num,
            "Precision": precision,
            "Recall": recall,
            "SMATCH": best_f_score
        }


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.critical(args)
    state = SmatchScript(**vars(args))

    if args.one_line == 'both':
        prod_amrs = [x.strip() for x in open(args.f[0], 'r')]
        gold_amrs = [x.strip() for x in open(args.f[1], 'r')]
    elif args.one_line == 'prod':
        prod_amrs = [x.strip() for x in open(args.f[0], 'r')]
        gold_amrs = SmatchScript.get_amr_line(args.f[1])
    elif args.one_line == 'gold':
        prod_amrs = SmatchScript.get_amr_line(args.f[0])
        gold_amrs = SmatchScript.get_amr_line(args.f[1])
    elif args.one_line == 'no':
        prod_amrs = SmatchScript.get_amr_line(args.f[0])
        gold_amrs = SmatchScript.get_amr_line(args.f[1])

    logger.info((len(gold_amrs), len(prod_amrs)))

    if len(gold_amrs) > len(prod_amrs):
        logger.error("Error: File 1 has less AMRs than file 2")
        raise ValueError
    if len(prod_amrs) > len(gold_amrs):
        logger.error("Error: File 2 has less AMRs than file 1")
        raise ValueError

    from tqdm import tqdm
    for cur_amr1, cur_amr2 in tqdm(list(zip(prod_amrs, gold_amrs)), disable=False):
        state.process_instance(cur_amr1, cur_amr2)
        print(state.total_match_num, state.total_test_num, state.total_gold_num)
        # state.report()

    state.report()


if __name__ == "__main__":
    main()
