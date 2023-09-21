from hstest import *


ANSWER = ["Made split: Sex is 0", "Made split: Pclass is 2", "Made split: Pclass is 3"]


class RecSplitTest(StageTest):

    @dynamic_test
    def test(self):
        pr = TestedProgram()
        pr.start()
        if not pr.is_waiting_input():
            raise WrongAnswer("You program should input the path to the file")
        output = pr.execute("test/data_stage3.csv").strip()
        res = output.split("\n")
        res = [x.strip() for x in res]
        if len(res) != 3:
            raise WrongAnswer("Wrong number of splits. Your function should make a split three times (on given data)."
                              "\nIt may be useful to check the traversal order in the recursive split function: go to the left child first.")
        if res[0] != ANSWER[0]:
            raise WrongAnswer("Wrong first log message. Correct message template: 'Made split: Sex is 1'."
                              "\nIt may be useful to check the traversal order in the recursive split function: go to the left child first.")
        if res[1] != ANSWER[1]:
            raise WrongAnswer("Wrong second log message. Correct message template: 'Made split: Sex is 1'."
                              "\nIt may be useful to check the traversal order in the recursive split function: go to the left child first.")
        if res[2] != ANSWER[2]:
            raise WrongAnswer("Wrong third log message. Correct message template: 'Made split: Sex is 1'."
                              "\nIt may be useful to check the traversal order in the recursive split function: go to the left child first.")
        return CheckResult.correct()


if __name__ == '__main__':
    RecSplitTest().run_tests()