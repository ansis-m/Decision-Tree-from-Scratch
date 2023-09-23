from dataclasses import dataclass

import numpy as np
import pandas as pd
from typing import Union
from graphviz import Digraph
from sklearn.metrics import confusion_matrix

TRAIN_FILE_PATH = ".\\test\\data_stage9_train.csv"
TEST_FILE_PATH = ".\\test\\data_stage9_test.csv"
MAX_GINI = 1.0
MINIMUM_SAMPLES = 74


@dataclass
class DecisionNode:
    feature_index: int
    value: str
    left: 'Union[DecisionNode, LeafNode]'
    right: 'Union[DecisionNode, LeafNode]'
    name: str
    gini: float
    float_type: bool


@dataclass
class LeafNode:
    prediction: int


def gini_impurity(array: pd.Series) -> float:
    array_len = len(array)
    if array_len == 0:
        return 0
    array_sum = np.sum(array)
    return 1 - (array_sum / array_len) ** 2 - (1 - array_sum / array_len) ** 2


def not_redundant(node: pd.DataFrame) -> bool:
    features = node.iloc[:, :-1]
    return not (features.nunique() == 1).all()


def construct_tree(df: pd.DataFrame):
    # print(df)
    outcome: pd.Series = df.iloc[:, -1]
    rows = len(df)
    gini = MAX_GINI
    index = -1
    value = ""

    def _best_weighted_gini_for_column(index: int) -> tuple:
        column: pd.Series = df.iloc[:, index]
        column_gini: float = MAX_GINI
        column_value: str = ""
        unique = column.unique()

        for entry in unique:
            this_gini = get_weighted_gini(entry, column, column.dtype == 'float64')
            if this_gini < column_gini:
                column_gini = this_gini
                column_value = entry

        return column_gini, column_value

    def print_results():
        print("Made split: {} is {}".format(
            df.columns[index],
            value,
        ))

    def get_weighted_gini(entry, column, float_type=False) -> float:
        if float_type:
            matching = outcome[column <= entry]
            not_matching = outcome[column > entry]
        else:
            matching = outcome[column == entry]
            not_matching = outcome[column != entry]

        gini_matching = gini_impurity(matching)
        gini_not_matching = gini_impurity(not_matching)
        return len(matching) / rows * gini_matching + len(not_matching) / rows * gini_not_matching

    def parse_nodes(node):
        if gini_impurity(node[node.columns[-1]]) != 0 and node.shape[0] > MINIMUM_SAMPLES and not_redundant(node):
            return construct_tree(node)
        else:
            return LeafNode(node[node.columns[-1]].mode()[0])

    for i in range(0, df.shape[1] - 1):
        column_gini, column_value = _best_weighted_gini_for_column(i)
        if column_gini < gini:
            gini = column_gini
            index = i
            value = column_value

    # print_results()

    float_type = df.iloc[:, index].dtype == 'float64'

    left_node = df[df.iloc[:, index] <= value] if float_type else df[df.iloc[:, index] == value]
    right_node = df[df.iloc[:, index] > value] if float_type else df[df.iloc[:, index] != value]

    left_tree, right_tree = parse_nodes(left_node), parse_nodes(right_node)
    return DecisionNode(index, value, left_tree, right_tree, df.columns[index], gini, float_type)


def visualize_tree(node, df, parent_name='', graph=None):
    if graph is None:
        graph = Digraph()

    if isinstance(node, LeafNode):
        graph.node(f'{id(node)}', label=f"Prediction: {node.prediction}")
    else:
        graph.node(f'{id(node)}', label=f"Feature {df.columns[node.feature_index]} == {node.value}?")
        visualize_tree(node.left, df, f'{id(node)}', graph)
        graph.edge(f'{id(node)}', f'{id(node.left)}', label='True')
        visualize_tree(node.right, df, f'{id(node)}', graph)
        graph.edge(f'{id(node)}', f'{id(node.right)}', label='False')

    return graph


def predict(row, node, index):
    if index != -1:
        # print("Prediction for sample # {}".format(index))
        pass
    if isinstance(node, LeafNode):
        # print("\tPredicted label: {}".format(node.prediction))
        return node.prediction
    elif not node.float_type and row.iloc[node.feature_index] == node.value:
        # print("\tConsidering decision rule on feature {} with value {}".format(node.name, node.value))
        return predict(row, node.left, -1)
    elif node.float_type and row.iloc[node.feature_index] <= node.value:
        # print("\tConsidering decision rule on feature {} with value {}".format(node.name, node.value))
        return predict(row, node.left, -1)
    else:
        # print("\tConsidering decision rule on feature {} with value {}".format(node.name, node.value))
        return predict(row, node.right, -1)


def main():
    TRAIN_FILE_PATH, TEST_FILE_PATH = input().split(" ")
    df = pd.read_csv(TRAIN_FILE_PATH)
    df.set_index(df.columns[0], inplace=True)
    df.index.name = 'index'
    # print(df)
    tree: DecisionNode = construct_tree(df)

    test_df: pd.DataFrame = pd.read_csv(TEST_FILE_PATH)
    test_df.set_index(test_df.columns[0], inplace=True)
    predictions = test_df.apply(lambda row: predict(row, tree, row.name), axis=1)
    outcome: pd.Series = test_df.iloc[:, -1]

    matrix = confusion_matrix(outcome, predictions)
    print(round(matrix[1][1] / (matrix[1][0] + matrix[1][1]), 3),
          round(matrix[0][0] / (matrix[0][0] + matrix[0][1]), 3))

    # graph = visualize_tree(tree, df)
    # graph.view()


if __name__ == "__main__":
    main()
