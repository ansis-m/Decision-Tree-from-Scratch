from hstest import *

ANSWER = [0.5, 0.42]


class FuncTest(StageTest):

    @dynamic_test
    def test(self):
        pr = TestedProgram()
        pr.start()
        if not pr.is_waiting_input():
            raise WrongAnswer("You program should input the contents of the node.")
        pr.execute("1 1 1 0 0 0 0 0 1 1")
        if not pr.is_waiting_input():
            raise WrongAnswer("You program should input two sets of labels after the split.")
        pr.execute("1 1 1 0")
        if not pr.is_waiting_input():
            raise WrongAnswer("You program should input two sets of labels after the split. Only the first set had been taken, the second one should be input as well.")
        output = pr.execute("0 0 0 0 1 1").strip()
        res = output.split()
        if len(res) != 2:
            raise WrongAnswer(f"You should print two values split with a space. Found {len(res)}.")
        res = [round(float(x), 2) for x in res]
        if res[0] != ANSWER[0]:
            raise WrongAnswer("Wrong answer of the function computing Gini impurity (the first value).")
        if res[1] != ANSWER[1]:
            raise WrongAnswer("Wrong answer of the function computing weighted Gini impurity (the second value).")
        return CheckResult.correct()


if __name__ == '__main__':
    FuncTest().run_tests()
