#!/usr/bin/env python
# -*- coding: utf8 -*-

"""Script that tests given seq2seq model on given test data, also restoring and wikifying the produced AMRs

Input should either be a produced AMR -file or a folder to traverse. Outputs .restore, .pruned, .coref and .all files"""

import os
import logging
import argparse

from .amr_utils import valid_amr, get_default_amr

logger = logging.getLogger('amr_postprocessing')

def countparens(text):
    """ proper nested parens counting """
    currcount = 0
    for i in text:
        if i == "(":
            currcount += 1
        elif i == ")":
            currcount -= 1
            if currcount < 0:
                return False
    return currcount == 0


def create_arg_parser():
    ### If using -fol, -f and -s are directories.
    # In that case the filenames of the sentence file and output
    # file should match (except extension)
    ### If not using -fol, -f and -s are directories ###

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', required=True, help="File or folder to be post-processed")
    parser.add_argument('-s', default='', help="Sentence file or folder, necessary for Wikification")

    parser.add_argument('-fol', action='store_true', help="Whether -f is a folder")
    parser.add_argument('-sent_ext', default='.sent',
                        help="Sentence file, necessary for Wikification - only needed when doing single file")
    parser.add_argument('-out_ext', default='.seq.amr', help="Output directory when doing a file")
    parser.add_argument('-t', default=16, type=int, help="Maximum number of parallel threads")

    parser.add_argument('-c', default='dupl', action='store', choices=['dupl', 'index', 'abs'],
                        help='How to handle coreference - input was either duplicated/indexed/absolute path')
    parser.add_argument('-no_wiki', action='store_true', help='Not doing Wikification, since it takes a long time')
    args = parser.parse_args()

    return args


def check_valid(line):
    if not valid_amr(line):
        logger.warning(f'Error or warning, write default')
        default_amr = get_default_amr()
        return default_amr
        all_amrs.append(default_amr)  ## add default when error
    return line


def add_coreference_item(item):
    from .restore_duplicate_coref import process_item

    item = process_item(item)
    return item


def do_pruning_item(item):
    from .prune_amrs import prune_item
    return prune_item(item)


def restore_amr_item(line):
    from .restore_amr import process_item
    return process_item(line)


def process_item(item, coref=True):
    item = restore_amr_item(item)
    if not coref:
        item = do_pruning_item(item)
    else:
        item = add_coreference_item(item)

    item = check_valid(item)
    return item


def main(predictions_path):

    if not os.path.getsize(predictions_path):
        return

    with open(predictions_path, 'r', encoding='utf-8') as fd:
        for line in fd:
            item = process_item(line)
            print(item)


if __name__ == "__main__":
    args = create_arg_parser()

    main(predictions_path=args.f)
