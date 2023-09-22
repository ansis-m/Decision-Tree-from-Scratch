from hstest import *


ANSWER = ["Made split: Sex is 0", "Made split: Pclass is 2",
          "Made split: Pclass is 1", "Made split: Pclass is 3"]


class TreeTest(StageTest):

    @dynamic_test
    def test(self):
        pr = TestedProgram()
        pr.start()
        if not pr.is_waiting_input():
            raise WrongAnswer("You program should input the path to the file")
        output = pr.execute("test/data_stage4.csv").strip()
        res = output.split("\n")
        res = [x.strip() for x in res]
        if len(res) != 4:
            raise WrongAnswer("Wrong number of splits. Your model should make a split four times (on given data).")
        for i in range(4):
            if res[i] != ANSWER[i]:
                raise WrongAnswer(f"Wrong log message on line {i+1}. Correct message template: 'Made split: Sex is 1'."
                                  "\nIt may be useful to check the traversal order in the recursive split function: go to the left child first.")
        return CheckResult.correct()


if __name__ == '__main__':
    TreeTest().run_tests()
