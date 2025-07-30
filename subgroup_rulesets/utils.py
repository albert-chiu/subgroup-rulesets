import numpy as np
import operator
from bisect import bisect_left


def extract_rules(tree, feature_names):
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    # get ids of child nodes
    idx = np.argwhere(left == -1)[:,0]

    def recurse(left, right, child, lineage=None):
        if lineage is None:
            lineage = []
        if child in left:
            parent = np.where(left == child)[0].item()
            suffix = '_neg'
        else:
            parent = np.where(right == child)[0].item()
            suffix = ''

        #           lineage.append((parent, split, threshold[parent], features[parent]))
        lineage.append((features[parent].strip()+suffix))

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)
    rules = []
    for child in idx:
        rule = []
        if child !=0:
            for node in recurse(left, right, child):
                rule.append(node)
            rules.append(rule)
    return rules

def accumulate(iterable, func=operator.add):
    'Return running totals'
    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = func(total, element)
        yield total

def find_lt(a, x):
    """ Find rightmost value less than x"""
    i = bisect_left(a, x)
    if i:
         return int(i-1)
    print('in find_lt,{}'.format(a))
    raise ValueError
    
def get_stats(df, Y, rules):
    covered = np.zeros(len(df))
    for rule in rules:
        val = np.array([not condition.endswith('_neg') for condition in rule]).astype(int)
        rule_cleaned = [condition.replace('_neg', '') for condition in rule]
        covered = covered + np.all(df[rule_cleaned]==val, axis=1)
    return np.mean(Y[covered > 0]), np.sum([x > 0 for x in covered])