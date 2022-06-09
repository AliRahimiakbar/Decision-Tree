import pandas as pd
import math
import random
from sklearn.model_selection import train_test_split

Name = "Ali RahimiAkbar"
Student_Number = "99101621"


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        """
        Class for storing Decision Tree as a binary-tree
        Inputs:
        - feature: Name of the the feature based on which this node is split
        - threshold: The threshold used for splitting this subtree
        - left: left Child of this node
        - right: Right child of this node
        - value: Predicted value for this node (if it is a leaf node)
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        if self.value is None:
            return False
        return True


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        """
        Class for implementing Decision Tree
        Attributes:
        - max_depth: int
            The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until
            all leaves contain less than min_samples_split samples.
        - min_num_samples: int
            The minimum number of samples required to split an internal node
        - root: Node
            Root node of the tree; set after calling fit.
        """
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def is_splitting_finished(self, depth, num_class_labels, num_samples):
        """
        Criteria for continuing or finishing splitting a node
        Inputs:
        - depth: depth of the tree so far
        - num_class_labels: number of unique class labels in the node
        - num_samples: number of samples in the node
        :return: bool
        """
        if depth >= self.max_depth or num_samples <= self.min_samples_split or num_class_labels == 1:
            return True
        return False

    def split(self, X, y, feature, threshold):
        """
        Splitting X and y based on value of feature with respect to threshold;
        i.e., if x_i[feature] <= threshold, x_i and y_i belong to X_left and y_left.
        Inputs:
        - X: Array of shape (N, D) (number of samples and number of features respectively), samples
        - y: Array of shape (N,), labels
        - feature: Name of the the feature based on which split is done
        - threshold: Threshold of splitting
        :return: X_left, X_right, y_left, y_right
        """
        l_index = X[feature] <= threshold
        r_index = X[feature] > threshold
        return X[l_index], X[r_index], y[l_index], y[r_index]

    def entropy(self, y):
        """
        Computing entropy of input vector
        - y: Array of shape (N,), labels
        :return: entropy of y
        """
        if len(y) == 0:
            return 0
        entropy = 0
        zero = 0
        one = 0
        for i in y.target:
            if i == 0:
                zero += 1
            else:
                one += 1
        zero = zero / len(y)
        one = one / len(y)
        if zero != 0:
            entropy -= zero * math.log(zero, 2)
        if one != 0:
            entropy -= one * math.log(one, 2)
        return entropy

    def information_gain(self, X, y, feature, threshold):
        """
        Returns information gain of splitting data with feature and threshold.
        Hint! use entropy of y, y_left and y_right.
        """
        X_left, X_right, y_left, y_right = self.split(X, y, feature, threshold)
        entropy_y = self.entropy(y)
        entropy_y_r = self.entropy(y_right)
        entropy_y_l = self.entropy(y_left)
        p = len(X_left) / len(X)
        entropy_all = p * entropy_y_l + (1 - p) * entropy_y_r
        return entropy_y - entropy_all

    def best_split(self, X, y):
        """
        Used for finding best feature and best threshold for splitting
        Inputs:
        - X: Array of shape (N, D), samples
        - y: Array of shape (N,), labels
        :return:
        """
        features = list(X.columns.values)
        random.Random(2).shuffle(features)
        feature_holder = None
        thresholds_holder = None
        max_gain = -1
        for feature in features:
            thresholds = list(X[feature])
            for threshold in thresholds:
                information_gain_temp = self.information_gain(X, y, feature, threshold)
                if max_gain < information_gain_temp:
                    max_gain = information_gain_temp
                    feature_holder = feature
                    thresholds_holder = threshold
        return feature_holder, thresholds_holder

    def build_tree(self, X, y, depth=0):
        """
        Recursive function for building Decision Tree.
        - X: Array of shape (N, D), samples
        - y: Array of shape (N,), labels
        - depth: depth of tree so far
        :return: root node of subtree
        """
        is_finished = self.is_splitting_finished(depth, len(X.columns), len(X))
        if is_finished:
            return None
        feature, threshold = self.best_split(X, y)
        X_left, X_right, y_left, y_right = self.split(X, y, feature, threshold)

        left_tree = self.build_tree(X_left, y_left, depth + 1)
        right_tree = self.build_tree(X_right, y_right, depth + 1)

        value = None
        if right_tree is None or left_tree is None:
            if len(y[y['target'] == 1]) > len(y[y['target'] == 0]):
                value = 1
            else:
                value = 0

        return Node(feature, threshold, left_tree, right_tree, value)

    def fit(self, X, y):
        """
        Builds Decision Tree and sets root node
        - X: Array of shape (N, D), samples
        - y: Array of shape (N,), labels
        """
        self.root = self.build_tree(X, y, 0)

    def predict(self, X):
        """
        Returns predicted labels for samples in X.
        :param X: Array of shape (N, D), samples
        :return: predicted labels
        """
        predict_value = []
        for data_index in list(X.index):
            data = X.loc[data_index]
            label_node = self.root
            while True:
                if Node.is_leaf(label_node):
                    predict_value.append(label_node.value)
                    break
                elif data[label_node.feature] <= label_node.threshold:
                    label_node = label_node.left
                else:
                    label_node = label_node.right
        return predict_value


# import data
data_csv = pd.read_csv("breast_cancer.csv")

# Split your data to train and validation sets
x_train, x_val, y_train, y_val = train_test_split(data_csv.drop('target', axis=1), data_csv[['target']], test_size=0.70,
                                                  random_state=5)

# Tune your hyper-parameters using validation set

# Train your model with hyper-parameters that works best on validation set

max_depths = [1, 2, 3, 4, 5]
min_samples_splits = [1, 2, 3]

best_max_depth = None
best_min_samples_split = None
best_accuracy = 0
best_tree = None
for max_depth in max_depths:
    for min_samples_split in min_samples_splits:
        test_tree = DecisionTree(max_depth, min_samples_split)
        test_tree.fit(x_train, y_train)
        y_val_pred = test_tree.predict(x_val)
        same_result = 0
        val_holder = list(y_val['target'])
        for i in range(len(y_val_pred)):
            if y_val_pred[i] == val_holder[i]:
                same_result += 1
        accuracy = same_result / len(y_val_pred)
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_max_depth = max_depth
            best_min_samples_split = min_samples_split
            best_tree = test_tree

print(best_accuracy)

# Predict test set's labels

test_csv = pd.read_csv("test.csv")

test_predict = best_tree.predict(test_csv)
data_frame = pd.DataFrame(test_predict, columns=['target'])
data_frame.to_csv('output.csv', index=False)
