import string
import numpy as np
import pandas as pd

FILE_PATH = ".\\test\\data_stage2.csv"


def gini_impurity(array) -> float:
    array_len = len(array)
    if array_len == 0:
        return 0
    array_sum = np.sum([int(x) for x in array])
    return 1 - (array_sum / array_len) ** 2 - (1 - array_sum / array_len) ** 2


class Gini:
    def __init__(self, file_path: string):
        self.df: pd.DataFrame = pd.read_csv(file_path)
        self.df.set_index(self.df.columns[0], inplace=True)
        self.df.index.name = 'index'
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
        print("{} {} {} {} {}".format(
            self.result,
            self.df.columns[self.index],
            self.value,
            self.df[self.df.iloc[:, self.index] == self.value].index.tolist(),
            self.df[self.df.iloc[:, self.index] != self.value].index.tolist()

        ))

    def get_weighted_gini(self, entry, column) -> float:
        matching = self.outcome[column == entry]
        not_matching = self.outcome[column != entry]

        gini_matching = gini_impurity(matching)
        gini_not_matching = gini_impurity(not_matching)
        return len(matching) / self.rows * gini_matching + len(not_matching) / self.rows * gini_not_matching


def main():
    FILE_PATH = input()
    gini = Gini(FILE_PATH)
    gini.print_results()


if __name__ == "__main__":
    main()
