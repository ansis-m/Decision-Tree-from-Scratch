from hstest import *
import ast

ANSWER = [0.0, 'Sex', 0, [0, 4, 5, 6, 7], [1, 2, 3, 8, 9]]


class SplitTest(StageTest):

    @dynamic_test
    def test(self):
        pr = TestedProgram()
        pr.start()
        if not pr.is_waiting_input():
            raise WrongAnswer("You program should input the path to the file")
        output = pr.execute("test/data_stage2.csv").strip().replace(", ", ",")
        output = [x.strip() for x in output.split()]
        if len(output) != 5:
            raise WrongAnswer("Your answer should contain five items split by a space: a float, a string, an integer, "
                              "and two lists.")

        # checking the first item
        try:
            user_gini = float(output[0])
        except Exception:
            raise WrongAnswer("The first value should be a float.")
        if user_gini != ANSWER[0]:
            raise WrongAnswer("Wrong Gini score (the first item in your answer).")

        # checking the second item
        if output[1] != ANSWER[1]:
            raise WrongAnswer("Wrong threshold feature (the second item in your answer).")

        # checking the third item
        try:
            user_threshold = int(output[2])
        except Exception:
            raise WrongAnswer("The third value should be an integer.")
        if user_threshold != ANSWER[2]:
            raise WrongAnswer("Wrong threshold value (the third item in your answer).")

        # checking the forth and the fifth item
        for i, ans, node in zip([0, 1], output[3:], ["left", "right"]):
            index_from = ans.find('[')
            index_to = ans.find(']')
            list_str = ans[index_from: index_to + 1]
            try:
                user_list = ast.literal_eval(list_str)
            except Exception:
                return CheckResult.wrong(f"Seems that the {4 + i}th item of your output is in wrong format. The list is expected.")

            if not isinstance(user_list, list):
                return CheckResult.wrong(f'Print the {4 + i}th item as a list')
            if user_list != ANSWER[3+i]:
                raise WrongAnswer(f"Wrong list of {node} node indexes (the {4 + i}th item in your answer).")

        return CheckResult.correct()


if __name__ == '__main__':
    SplitTest().run_tests()
