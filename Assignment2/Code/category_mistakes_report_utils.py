import collections
import inspect
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import time

import Assignment2Support as utils
import EvaluationsStub as es
import LogisticRegressionModel as lgm
import features_by_frequency as fbf
import features_by_mi as fbm


# File/Folder path
kDataPath = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), r"Data/SMSSpamCollection")

report_path = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), r"Report")

fn_file = os.path.join(report_path, 'category_mistake_false_negatives.md')
fp_file = os.path.join(report_path, 'category_mistake_false_positives.md')

def get_most_common_words(fname, N=10):

    count = collections.Counter()
    with open(fname, 'r') as f:
        for line in f.readlines():
            if not line.strip().startswith('|'):
                continue
            cols = line.split('|')
            for word in cols[2].split(' '):
                count[word] += 1
    
    return count.most_common(N)

def lengthy(fname, L=40):

    cnt = 0
    with open(fname, 'r') as f:
        for line in f.readlines():
            if not line.strip().startswith('|'):
                continue
            if len(line.strip()) > L:
                cnt += 1
    return cnt

def has_uppers(fname, U=3):
    """Is the message has all-capital words more than U?"""
    cnt = 0
    with open(fname, 'r') as f:
        for line in f.readlines():
            if not line.strip().startswith('|'):
                continue
            cols = line.split('|')
            upper_cnt = 0
            for word in cols[2].split(' '):
                if word.isupper():
                    upper_cnt += 1
            if upper_cnt > U:
                cnt += 1
    return cnt

def contains(line, pattern):
    if pattern in line or re.search(pattern, line):
        return 1
    else:
        0

# category_funcs
def has_url(line):
    return contains(line, '(www|http)')


def lengthy_line(line, N=40):
    if len(line) > N:
        return 1
    return 0

def many_uppers(line, U=3):
    upper_cnt = 0
    for word in line.split(' '):
        if word.isupper():
            upper_cnt += 1
    if upper_cnt > U:
        return 1
    return 0

def has_reply(line):
    return contains(line, 'reply')

def has_call_nums(line):
    return contains(line, r'(call \d+|text \d+)|reply')

def has_dots(line):
    return contains(line, r'\.\.')

def has_lower_i(line):
    return contains(line, ' i ')

def very_long(line):
    return lengthy_line(line, 120)

def has_exlamations(line):
    return contains(line, '.*!.*!')

category_funcs = [has_call_nums, many_uppers, has_url, lengthy_line]

def categorize(fname, category_funcs=category_funcs, w=20):

    count = collections.Counter()
    categories = [c.__name__ for c in category_funcs]
    table = '* Categorized Message'
    table += '\n  |{}|{}|{}|'.format('Message Index'.center(w), 'Type'.center(w), 'Category'.center(w))
    table += '\n  |{}|{}|{}|'.format('-'*w,'-'*w,'-'*w)
    with open(fname, 'r') as f:
        l_cnt = 0
        cnt = 0
        msg_index = 0
        selected = []
        lines = [x for x in f.readlines()]
        for lo, line in enumerate(lines):
            if not line.strip().startswith('|'):
                selected.append(lo)
                continue
            if 'Probabilities' in line or '|-|-|' in line:
                selected.append(lo)
                continue
            
            for cf in category_funcs:
                val = cf(line.strip())
                if val == 1:
                    cf_name = cf.__name__
                    count[cf_name] += 1
                    cnt += 1
                    selected.append(lo)
                    table += '\n  |{}|{}|{}|'.format(str(msg_index).center(w), str(categories.index(cf_name)).center(w), cf_name.center(w))
                    break
            msg_index += 1
            l_cnt += 1
        count['msci.'] += l_cnt - cnt
        missing_index = list(set([x for x in range(len(lines))]) - set(selected))
        for m in missing_index:
            print(lines[m])
    print(table)
    return count.most_common(len(category_funcs) + 1)

if __name__ == "__main__":
    N = 10
    L = 100
    U = 3
    fn_mcw = get_most_common_words(fn_file, N)
    print(fn_mcw)
    fn_lengthy = lengthy(fn_file, L)
    print(fn_lengthy)
    fn_uppers = has_uppers(fn_file, U)
    print(fn_uppers)
    print(categorize(fn_file))
    
    print('+' * 80)
    L = 2000
    
    fp_mcw = get_most_common_words(fp_file, N)
    print(fp_mcw)
    fp_lengthy = lengthy(fp_file, L)
    print(fp_lengthy)
    fp_uppers = has_uppers(fp_file, U)
    print(fp_uppers)
    category_funcs = [has_exlamations, has_lower_i, has_dots, very_long]
    print(categorize(fp_file, category_funcs))
    
    # https://medium.com/greyatom/performance-metrics-for-classification-problems-in-machine-learning-part-i-b085d432082b