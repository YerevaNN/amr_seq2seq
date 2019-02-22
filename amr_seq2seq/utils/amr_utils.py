#!/usr/bin/env python
# -*- coding: utf8 -*-

"""General utils and AMR specific utils"""

import os
import json
import logging


logger = logging.getLogger('amr_postprocessing')


def get_default_amr():
    default = '(w / want-01 :ARG0 (b / boy) :ARG1 (g / go-01 :ARG0 b))'
    return default


def write_to_file(lst, file_new):
    with open(file_new, 'w', encoding='utf-8') as out_f:
        for line in lst:
            out_f.write(line.strip() + '\n')
    out_f.close()


def get_files_by_ext(direc, ext):
    """Function that traverses a directory and returns all files that match a certain extension"""

    return_files = []
    for root, dirs, files in os.walk(direc):
        for f in files:
            if f.endswith(ext):
                return_files.append(os.path.join(root, f))

    return return_files


def tokenize_line(line):
    new_l = line.replace('(', ' ( ').replace(')', ' ) ')
    return " ".join(new_l.split())


def reverse_tokenize(new_line):
    while ' )' in new_line or '( ' in new_line:  # restore tokenizing
        new_line = new_line.replace(' )', ')').replace('( ', '(')

    return new_line


def load_dict(d):
    """Function that loads json dictionaries"""

    with open(d,
              'r',
              encoding='utf-8') as in_f:  # load reference dict (based on training data) to settle disputes based on frequency
        dic = json.load(in_f)
    in_f.close()

    return dic


def add_to_dict(d, key, base):
    """Function to add key to dictionary, either add base or start with base"""

    if key in d:
        d[key] += base
    else:
        d[key] = base

    return d


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


def valid_amr(amr_text):
    from . import amr

    if not countparens(amr_text):  ## wrong parentheses, return false
        return False

    try:
        theamr = amr.AMR.parse_AMR_line(amr_text)
        if theamr is None:
            return False
            logger.error(f"MAJOR WARNING: couldn't build amr out of {amr_text} using smatch code")
        else:
            return True

    except (AttributeError, Exception) as e:
        logger.error(e)
        return False

    return True
