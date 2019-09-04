#!/usr/bin/env python
"""
  Decision tree impl using Numpy

  Hz, 2019-5
"""

import numpy as np
import math


class TreeEnsemble(object):

    def __init__(self, n_trees, sample_sz, min_leaf=5):
        np.random.seed(42)
        self.n_trees = n_trees
        self.sample_sz = sample_sz
        self.min_leaf = min_leaf
        
    def fit(self, x, y, x_title=None):
        self.x, self.y, self.x_title = x, y, x_title
        # use Bagging to create a forest
        self.trees = [self.create_tree() for i in range(self.n_trees)]  

    def create_tree(self):
        # create trees by random dataset
        tree_sz = min(len(self.y), self.sample_sz)
        idxs = np.random.permutation(len(self.y))[:tree_sz]  
        return DecisionTree(self.x[idxs], 
                            self.y[idxs], 
                            idxs=np.arange(tree_sz), 
                            x_title=self.x_title, 
                            min_leaf=self.min_leaf)

    def predict(self, x):
        return np.mean([tree.predict(x) for tree in self.trees], axis=0)


def std_agg(cnt, s1, s2):
    # add 1e-10 in case agg is negative due to platform float accuracy
    agg = 1e-10 + (s2 / cnt) - (s1 / cnt) ** 2
    assert agg >= 0, 'std_agg: s2:{} s1:{} cnt:{}'.format(s2,s1,cnt)
    return math.sqrt(agg)


class DecisionTree(object):

    def __init__(self, x, y, idxs, x_title=None, y_title=None, min_leaf=5):
        assert isinstance(x, np.ndarray)
        assert len(x.shape) == 2
        assert isinstance(y, np.ndarray) and len(y) >= 22  # lucky number 22
        assert len(x) == len(y)
        self.x, self.y = x, y
        self.x_title, self.y_title = x_title, y_title
        self.min_leaf = min_leaf
        self.n = len(idxs)  # number of samples         
        self.idxs = idxs # np.arange(len(self.y))
        self.c = x.shape[1]  # x data columns
        self.value = np.mean(y[idxs])  # actual prediction result
        # set initial score to inf. lower number means better score because it is composed by deviation.
        self.score = float('inf')
        # initially, find a X variable (column) to create the tree
        self.find_varsplit()

    def find_varsplit(self):
        for i in range(self.c):
            self.find_better_varsplit(i)
        if not self.is_leaf:
            x = self.split_col # 1d array
            lhs = np.nonzero(x <= self.split)[0] # indicies of non-zero in x
            rhs = np.nonzero(x > self.split)[0]
#             print('idxs:',len(self.idxs), 'lhs:',len(lhs), 'rhs:',len(rhs), len(self.x), len(self.y))
#             print(self.idxs)
#             print(lhs)
            self.lhs = DecisionTree(self.x, self.y, self.idxs[lhs], self.x_title)
            self.rhs = DecisionTree(self.x, self.y, self.idxs[rhs], self.x_title)

    def find_better_varsplit(self, var_idx):
        # keep finding the better X variable over random data set, then use this X variable as tree root node
        x, y = self.x[self.idxs, var_idx], self.y[self.idxs]  # x, y in 1-D array
        # sort samples to help calculating STD more efficient
        sort_idx = np.argsort(x)
        sort_x, sort_y = x[sort_idx], y[sort_idx]
        rhs_cnt, rhs_sum, rhs_sum2 = self.n, sort_y.sum(), (sort_y ** 2).sum()
        lhs_cnt, lhs_sum, lhs_sum2 = 0, 0, 0
        # iterate over each row (sample in order), and find the best tree split Xi (with least weighted variance)
        for i in range(self.n - self.min_leaf):
            xi, yi = sort_x[i], sort_y[i]
            rhs_cnt -= 1
            rhs_sum -= yi
            rhs_sum2 -= yi ** 2
            lhs_cnt += 1
            lhs_sum += yi
            lhs_sum2 += yi ** 2
            if i < self.min_leaf - 1 or xi == sort_x[i + 1]:
                continue

            lhs_std = std_agg(lhs_cnt, lhs_sum, lhs_sum2)
            rhs_std = std_agg(rhs_cnt, rhs_sum, rhs_sum2)
            curr_score = lhs_std * lhs_cnt + rhs_std * rhs_cnt # weighted variance
            if curr_score < self.score:
                self.var_idx = var_idx  # todo: not inited
                self.score = curr_score
                self.split = xi  # todo: not inited

    @property
    def split_name(self):
        return None if self.x_title is None else self.x_title[self.var_idx]

    @property
    def split_col(self):
        return self.x[self.idxs, self.var_idx]

    @property
    def is_leaf(self):
        return self.score == float('inf')

    def __repr__(self):
        s = 'n: {}; value: {}'.format(self.n, self.value)
        if not self.is_leaf:
            s += '; score: {}; split: {}; var: {}'.format(self.score, self.split, self.split_name)
        return s

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf:
            return self.value

        tree = self.lhs if xi[self.var_idx] <= self.split else self.rhs  # recursively iterate trees
        return tree.predict_row(xi)


if __name__ == "__main__":
    rng = np.random.RandomState(42)
    x = 10 * rng.rand(10000)
    y = np.sin(5.0 *x) + np.sin(0.5*x) + rng.randn(len(x))
    x = x.reshape(-1,1)
    ensemble = TreeEnsemble(1, 1000)
    ensemble.fit(x, y)
    # print(ensemble.trees[0])
    # print(ensemble.trees[0].lhs)
    # > python tree.py
    # n: 1000; value: 0.16209558918; score: 1195.33176578; split: 6.82671152463; var: None
    # n: 693; value: 0.700398222963; score: 831.983442498; split: 4.41833704802; var: None

    def traversal(tree):
        if tree.is_leaf:
            return
        print('lhs' if tree.value <= tree.split else 'rhs', tree)
        traversal(tree.lhs)
        traversal(tree.rhs)

    traversal(ensemble.trees[0])
    # todo: the tree is unbalance, improve it..

