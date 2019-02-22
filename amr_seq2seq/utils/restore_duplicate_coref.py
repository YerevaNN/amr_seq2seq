import re

import logging

from .amr_utils import valid_amr

logger = logging.getLogger('amr_postprocessing')


"""Script that adds coreference back in produced AMRs. It does this by simply replacing duplicate nodes by the reference to the variable of the first node.
Input needs to be in one-line format, with variables present.
Sample input:
(e / establish-01 :ARG1 (m / model :mod (i / innovate-01 :ARG1 (i2 / industry) :ARG1 (m2 / model) :ARG1 (i3 / innovate-01))))
Sample output:
(e / establish-01 :ARG1 (m / model :mod (i / innovate-01 :ARG1 (i2 / industry) :ARG1 m :ARG1 i)))"""


def process_var_line(line):
    """Function that processes line with a variable in it. Returns the string without
       variables and the dictionary with var-name + var - value"""

    var_list = []
    curr_var_name, curr_var_value = False, False
    var_value, var_name = '', ''
    skip_first = True

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
                if not var_list and skip_first:
                    skip_first = False  # skip first entry, but only do it once. We never want to refer to the full AMR.
                else:
                    add_var_value = var_value.strip().replace(')', '')
                    var_list.append([var_name.strip(), add_var_value])
            var_name = ''
            continue

        if curr_var_name:  # add to variable name
            var_name += ch
        elif curr_var_value:  # add to variable value
            var_value += ch

    var_list.append([var_name.strip(), var_value.strip().replace(')', '')])  # add last one

    for item in var_list:
        try:
            if not item[1].split()[-1].isdigit() and len(
                    item[1].split()) > 1:  # keep in :quant 5 as last one, but not ARG1: or :mod
                item[1] = " ".join(item[1].split()[0:-1])
        except:
            logger.warning(f'Small error, just ignore: {item}') # should not happen often, but strange, unexpected output is always possible

    return var_list


def process_item(line):
    var_list = process_var_line(line)  # get list of variables and concepts
    new_line = line

    for idx in range(len(var_list) - 1):
        for y in range(idx + 1, len(var_list)):
            if var_list[idx][1] == var_list[y][1]:  # match - we see a concept we already saw before
                replace_item = var_list[y][0] + ' / ' + var_list[y][1]  # the part that needs to be replaced
                if replace_item in line:

                    new_line_replaced = re.sub(r'\({0} / [^\(]*?\)'.format(var_list[y][0]), var_list[idx][0],
                                               new_line)  # coref matching, replace :ARG1 (var / value) by :ARG refvar

                    if new_line_replaced != new_line:  # something changed
                        if valid_amr(new_line_replaced):  # only replace if resulting AMR is valid
                            new_line = new_line_replaced

    return new_line.strip()
