import numpy as np
import pandas as pd
from typing import Union

TRAIN_FILE_PATH = ".\\test\\data_stage7.csv"
MAX_GINI = 1.0
MINIMUM_SAMPLES = 74


class DecisionNode:
    def __init__(self, feature_index: int, value: str, left: 'Union[DecisionNode, LeafNode]',
                 right: 'Union[DecisionNode, LeafNode]', name: str, gini: float):
        self.gini = gini
        self.feature_index = feature_index
        self.value = value
        self.left = left
        self.right = right
        self.name = name


class LeafNode:
    def __init__(self, prediction: int):
        self.prediction = prediction


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

    left_node = df[df.iloc[:, index] == value]
    right_node = df[df.iloc[:, index] != value]

    left_tree, right_tree = parse_nodes(left_node), parse_nodes(right_node)
    return DecisionNode(index, value, left_tree, right_tree, df.columns[index], gini)


def main():
    TRAIN_FILE_PATH = input()
    df = pd.read_csv(TRAIN_FILE_PATH)
    df.set_index(df.columns[0], inplace=True)
    df.index.name = 'index'
    tree: DecisionNode = construct_tree(df)

    print("{} {} {} {} {}".format(
        round(tree.gini, 3),
        tree.name,
        tree.value,
        df[df.iloc[:, tree.feature_index] <= tree.value].index.tolist() if isinstance(tree.value, float) else df[df.iloc[:, tree.feature_index] == tree.value].index.tolist(),
        df[df.iloc[:, tree.feature_index] > tree.value].index.tolist() if isinstance(tree.value, float) else df[df.iloc[:, tree.feature_index] != tree.value].index.tolist()))


if __name__ == "__main__":
    main()
