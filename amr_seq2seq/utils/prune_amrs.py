#!/usr/bin/env python
# -*- coding: utf8 -*-

"""Script that removes duplicate output from output AMRs. Most code is from best_amr_permutation.py.
    It removes nodes with same argument + concept under the same parent.
    Also removes nodes that occur three times or more, no matter the parent.

    Sample input:

    (e / establish-01 :ARG1 (m / model :mod (i / innovate-01 :ARG1 (i2 / industry) :ARG1 (i3 / industry) :ARG1 (i4 / industry))))

    Sample output:

    (e / establish-01 :ARG1 (m / model :mod (i / innovate-01 :ARG1 (i2 / industry))))

    ARG1 - industry node occurs 3 times and therefore gets pruned twice in this example."""

import re
import sys
import importlib

importlib.reload(sys)  # TODO WTF???

# from .amr_utils import write_to_file
from .best_amr_permutation import get_permutations, get_best_perm, create_final_line


# def create_arg_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-f", required=True, type=str, help="File with AMRs (one line)")
#     parser.add_argument("-cut_off", default=15, type=int, help="When to cut-off number of permutations")
#     args = parser.parse_args()
#
#     return args


def restore_variables_item(item):
    from .restore_amr import process_item
    return process_item(item)


# def restore_variables(f, filtered_amrs):
#     restored = []
#     for item in filtered_amrs:
#         item = restore_variables_item(item)
#         restored.append(item)
#     return restored

    # write_to_file(filtered_amrs, f + '.pruned_temp')  # write variable-less AMR to file
    #
    # os.system(f'python -m amr_seq2seq.utils.restore_amr -f {f}.pruned_temp -o {f}.pruned')  # restore here
    # os.system(f'rm {f}.pruned_temp')  # remove temp file again


def prune_item(line, cut_off=15):
    clean_line = re.sub(r'\([A-Za-z0-9-_~]+ / ', r'(', line).strip()  # delete variables

    if clean_line.count(':') > 1:  # only try to do something if we can actually permutate

        permutations, keep_string1, all_perms = get_permutations(clean_line, 1, '', [], 'prune',
                                                                 cut_off)  # get initial permutations
        keep_str = '(' + keep_string1
        final_string = get_best_perm(permutations, keep_str, '', keep_str, all_perms, 'prune',
                                     cut_off)  # prune duplicate output here

        add_to = " ".join(create_final_line(final_string).split())  # create final AMR line
        clean_line = " ".join(clean_line.split())

        return add_to
    else:
        return clean_line.strip()


# def prune_file(f):
#     """Prune input file for duplicate input"""
#
#     filtered_amrs = []
#     pruned = []
#     for line in open(f, 'r', encoding='utf-8'):
#         item = prune_item(line)
#         filtered_amrs.append(item)
#
#         item = restore_variables_item(item)
#         pruned.append(item)
#
#     return pruned
#
# if __name__ == '__main__':
#     args = create_arg_parser()
#     print('Pruning {0}'.format(args.f))
#     prune_file(args.f)
