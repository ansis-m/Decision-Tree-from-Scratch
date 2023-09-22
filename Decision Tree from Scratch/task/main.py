import numpy as np
import pandas as pd

FILE_PATH = ".\\test\\data_stage31.csv"


def gini_impurity(array) -> float:
    array_len = len(array)
    if array_len == 0:
        return 0
    array_sum = np.sum([int(x) for x in array])
    return 1 - (array_sum / array_len) ** 2 - (1 - array_sum / array_len) ** 2


def construct_tree(df: pd.DataFrame):
    #print(df)
    outcome: pd.Series = df.iloc[:, -1]
    rows = len(df)
    gini = 1
    index = -1
    value = ""

    def _best_weighted_gini_for_column(index: int) -> tuple:
        column: pd.Series = df.iloc[:, index]
        column_result: float = 1.0
        column_value: str = ""
        unique = column.unique()

        for entry in unique:
            this_result = get_weighted_gini(entry, column)
            if this_result < column_result:
                column_result = this_result
                column_value = entry

        return column_result, column_value

    def print_results():
        print("Made split: {} is {}".format(
            df.columns[index],
            value,
        ))
        left_node = df[df.iloc[:, index] == value]
        right_node = df[df.iloc[:, index] != value]
        if gini_impurity(left_node[left_node.columns[-1]]) != 0 and left_node.shape[0] > 1 and not_redundant(left_node):
            construct_tree(left_node)

        if gini_impurity(right_node[right_node.columns[-1]]) != 0 and right_node.shape[0] > 1 and not_redundant(
                right_node):
            construct_tree(right_node)

    def get_weighted_gini(entry, column) -> float:
        matching = outcome[column == entry]
        not_matching = outcome[column != entry]

        gini_matching = gini_impurity(matching)
        gini_not_matching = gini_impurity(not_matching)
        return len(matching) / rows * gini_matching + len(not_matching) / rows * gini_not_matching

    def not_redundant(left_node):
        features = left_node.iloc[:, :-1]
        return not (features.nunique() == 1).all()

    for i in range(0, df.shape[1] - 1):
        column_gini, column_value = _best_weighted_gini_for_column(i)
        if column_gini < gini:
            gini = column_gini
            index = i
            value = column_value

    print_results()


def main():
    FILE_PATH = input()
    df = pd.read_csv(FILE_PATH)
    df.set_index(df.columns[0], inplace=True)
    df.index.name = 'index'
    construct_tree(df)


if __name__ == "__main__":
    main()
