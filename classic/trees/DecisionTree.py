import numpy as np
import math


class Node:
    def __init__(self, feat_type = None, is_leaf=False):
        self.left = None
        self.right = None
        self.is_leaf = is_leaf
        self.feat_type = feat_type
        self.j = None
        self.t = None
        self.classes_l = None
        self.classes_r = None
        self.value = None

    def init(self):
        self.left = Node()
        self.right = Node()

    def __str__(self):
        return f"j: {self.j}\nt: {self.t}\nleaf: {self.is_leaf}\nvalue: {self.value}\n"

FTYPE_CATEGORICAL = 0
FTYPE_CONTINUOUS = 1

class DecisionTree():
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.depth = 0
        self.levels = set()
        self.tree = Node()
        self.dim = None

    
    def split_cont_feature(self, X, y, feat_ind, feat_val):
        ind_l = X[:, feat_ind] <= feat_val
        ind_r = X[:, feat_ind] >  feat_val
        Xl = X[ind_l, :]
        yl = y[ind_l]
        Xr = X[ind_r, :]
        yr = y[ind_r]
        return Xl, yl, Xr, yr

    
    def split_cat_feature(self, X, y, feat_ind, classes_l, classes_r):
        ind_l = np.isin(X[:, feat_ind], classes_l)
        ind_r = np.isin(X[:, feat_ind], classes_r)
        assert(classes_l)
        assert(classes_r)
        print(classes_l.shape)
        print(ind_l)
        Xl = X[ind_l, :]
        yl = y[ind_l]
        Xr = X[ind_r, :]
        yr = y[ind_r]
        return Xl, yl, Xr, yr

    
    def define_feature_type(self, features, threshold_ratio=3):
        if np.unique(features).shape[0] * threshold_ratio < features.shape[0]:
            return FTYPE_CATEGORICAL
        return FTYPE_CONTINUOUS

    
    def calc_max_gain_cat(self, X, y, feat_ind):
        classes = np.unique(X[:, feat_ind])
        n_classes = classes.shape[0]
        max_gain = -1.0
        best_classes_l = None
        best_classes_r = None
        for i in range(2 ** (n_classes - 1) - 1):
            classes_l = []
            classes_r = []
            accum = i + 1
            for j in range(n_classes):
                if (accum & 1 == 0):
                    classes_l.append[classes[j]]
                else:
                    classes_r.append[classes[j]]
                accum = accum >> 1
            gain = self.calc_gain(X, y, *self.split_cat_feature(X, y, feat_ind, classes_l, classes_r))
            if gain > max_gain:
                max_gain = gain
                best_classes_l = classes_l.copy()
                best_classes_r = classes_r.copy()
        assert(max_gain >= 0)
        return max_gain, best_classes_l, best_classes_r

    
    def calc_max_gain_cont(self, X, y, feat_ind):
        max_f = max(X[:, feat_ind])
        t = None
        max_gain = -1.0
        for feature_val in X[:, feat_ind]:
            if feature_val == max_f:
                continue
            gain = self.calc_gain(X, y, *self.split_cont_feature(X, y, feat_ind, feature_val))
            if gain > max_gain:
                max_gain = gain
                t = feature_val
        assert(max_gain >= 0)
        return max_gain, t


    def calc_gain(self, X, y, Xl, yl, Xr, yr):
        nXr = Xr.shape[0]
        nXl = Xl.shape[0]
        nX = X.shape[0]
        gain = nX * self.impurity(y) - nXr * self.impurity(yr) - nXl * self.impurity(yl)
        return gain

        
    def split_node(self, X, y, node, level):
        node.init()
        if X.shape[0] == 1:
            node.is_leaf = True
            node.value = y[0]
            return
        max_gain = -1.0
        for feature_ind in range(X.shape[1]):
            features = X[:, feature_ind]
            feat_type = self.define_feature_type(features)
            gain_info = (self.calc_max_gain_cont(X, y, feature_ind) 
                          if feat_type == FTYPE_CONTINUOUS
                          else self.calc_max_gain_cat(X, y, feature_ind))
            if gain_info[0] > max_gain:
                max_gain = gain_info[0]
                if feat_type == FTYPE_CONTINUOUS:
                    node.j = feature_ind
                    node.t = gain_info[1]
                else:
                    node.j = feature_ind
                    node.classes_l = gain_info[1].copy()
                    node.classes_r = gain_info[2].copy()
                node.feat_type = feat_type
        self.levels.add(level)
        self.depth = len(self.levels)
        Xl, yl, Xr, yr = (self.split_cont_feature(X, y, node.j, node.t) 
                          if node.feat_type == FTYPE_CONTINUOUS 
                          else self.split_cat_feature(X, y, node.j, node.classes_l, node.classes_r))
        if level < self.max_depth and yl.shape[0] > 0 and yr.shape[0] > 0 and max_gain > 0:
            self.split_node(Xl, yl, node.left, level + 1)
            self.split_node(Xr, yr, node.right, level + 1)
        else:
            node.is_leaf = True
            node.value = self.calc_answer(y)

    
    def fit(self, X, y):
        self.dim = X.shape[1]
        self.split_node(X, y, self.tree, 0)

            
    def transform_vec(self, vec):
        assert(vec.shape[0] == self.dim)
        node = self.tree
        while not node.is_leaf:
            move = None
            if node.feat_type == FTYPE_CONTINUOUS:
                move = "left" if vec[node.j] <= node.t else "right"
            else:
                move = "left" if vec[node.j] in node.classes_l else "right" 
            node = node.left if move == "left" else node.right
        return node.value
            
        
    def predict(self, X):
        return np.apply_along_axis(self.transform_vec, 1, X)
    

class DecisionTreeRegressor(DecisionTree):
    def __init__(self, max_depth=5, impurity="mse"):
        super().__init__(max_depth)
        if impurity == "mse":
            self.impurity = self.impurity_mse
            self.calc_answer = self.calc_answer_mse
        elif impurity == "mae":
            self.impurity = self.impurity_mae
            self.calc_answer = self.calc_answer_mae
        else:
            raise ValueError

    
    def impurity_mse(self, y):
        return np.std(y) ** 2

    
    def calc_answer_mse(self, y):
        return np.mean(y)

    
    def impurity_mae(self, y):
        return np.mean(np.abs(y - np.median(y)))

    
    def calc_answer_mae(self, y):
        return np.median(y)
    

class DecisionTreeClassifier(DecisionTree):
    def __init__(self, max_depth=5, impurity="entropy"):
        super().__init__(max_depth)
        self.max_depth = max_depth
        if impurity == "entropy":
            self.impurity = self.impurity_entropy
        elif impurity == "gini":
            self.impurity = self.impurity_gini
        else:
            raise ValueError
        self.calc_answer = self.calc_max_proba
        self.classes = None

    def calc_probas(self, y):
        probas = []
        num_objs = y.shape[0]
        for cl in self.classes:
            probas.append(np.sum(y == cl) / num_objs)
        return probas

    def calc_max_proba(self, y):
        probas = self.calc_probas(y)
        return self.classes[np.argmax(np.array(probas))]
        
    def impurity_entropy(self, y):
        probas = self.calc_probas(y)
        imp = 0.0
        for p in probas:
            imp -= 0 if p == 0 else p * math.log(p)
        return imp

    def impurity_gini(self, y):
        probas = self.calc_probas(y)
        imp = 1.0
        for p in probas:
            imp -= p ** 2
        return imp

    def fit(self, X, y):
        self.dim = X.shape[1]
        self.classes = list(set(y))
        print(f"Classes: {self.classes}")
        self.split_node(X, y, self.tree, 0)
