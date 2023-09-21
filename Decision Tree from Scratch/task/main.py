import string
import numpy as np
import pandas as pd

FILE_PATH = ".\\test\\data_stage31.csv"


def gini_impurity(array) -> float:
    array_len = len(array)
    if array_len == 0:
        return 0
    array_sum = np.sum([int(x) for x in array])
    return 1 - (array_sum / array_len) ** 2 - (1 - array_sum / array_len) ** 2


class Gini:
    def __init__(self, df: pd.DataFrame):
        self.df: pd.DataFrame = df
        #print(self.df)
        self.outcome: pd.Series = self.df.iloc[:, -1]
        self.rows = len(self.df)
        self.result = 1
        self.index = -1
        self.value = ""

    def _run_splitting_function(self):
        for i in range(0, self.df.shape[1] - 1):
            gini, value = self._best_weighted_gini_for_column(i)
            if gini < self.result:
                self.result = gini
                self.index = i
                self.value = value
        return self

    def _best_weighted_gini_for_column(self, index: int) -> tuple:
        column: pd.Series = self.df.iloc[:, index]
        result: float = 1.0
        value: string = ""
        unique = column.unique()

        for entry in unique:
            this_result = self.get_weighted_gini(entry, column)
            if this_result < result:
                result = this_result
                value = entry

        return result, value

    def print_results(self):
        self._run_splitting_function()
        print("Made split: {} is {}".format(
            self.df.columns[self.index],
            self.value,
        ))
        left_node = self.df[self.df.iloc[:, self.index] == self.value]
        right_node = self.df[self.df.iloc[:, self.index] != self.value]
        if gini_impurity(left_node[left_node.columns[-1]]) != 0 and left_node.shape[0] > 1 and self.not_redundant(left_node):
            gini_left = Gini(left_node)
            gini_left.print_results()

        if gini_impurity(right_node[right_node.columns[-1]]) != 0 and right_node.shape[0] > 1 and self.not_redundant(right_node):
            gini_right = Gini(right_node)
            gini_right.print_results()


    def get_weighted_gini(self, entry, column) -> float:
        matching = self.outcome[column == entry]
        not_matching = self.outcome[column != entry]

        gini_matching = gini_impurity(matching)
        gini_not_matching = gini_impurity(not_matching)
        return len(matching) / self.rows * gini_matching + len(not_matching) / self.rows * gini_not_matching

    def not_redundant(self, left_node):
        features = left_node.iloc[:, :-1]
        return not (features.nunique() == 1).all()


def main():
    FILE_PATH = input()
    df = pd.read_csv(FILE_PATH)
    df.set_index(df.columns[0], inplace=True)
    df.index.name = 'index'

    gini = Gini(df)
    gini.print_results()


if __name__ == "__main__":
    main()
