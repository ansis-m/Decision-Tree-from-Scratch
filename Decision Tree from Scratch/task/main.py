import numpy as np


def get_input() -> np.array:
    return np.array([int(x) for x in input().split(" ")])


def get_impurity(array: np.array) -> float:
    return 1 - (np.sum(array) / len(array)) ** 2 - ((len(array) - np.sum(array)) / len(array)) ** 2


def weighted_gini(array1: np.array, array2: np.array) -> float:
    size = len(array1) + len(array2)
    return len(array1) / size * get_impurity(array1) + len(array2) / size * get_impurity(array2)


if __name__ == "__main__":
    node: np.array = get_input()
    split_1: np.array = get_input()
    split_2: np.array = get_input()
    impurity = get_impurity(node)
    weighted_impurity = weighted_gini(split_1, split_2)
    print("{:.5f}".format(impurity), "{:.5f}".format(weighted_impurity))
