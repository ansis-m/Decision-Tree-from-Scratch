from hstest import *


ANSWER = ['Prediction for sample # 0',
          'Considering decision rule on feature Sex with value 0',
          'Considering decision rule on feature SibSp with value 0',
          'Considering decision rule on feature Pclass with value 3',
          'Considering decision rule on feature Parch with value 0',
          'Predicted label: 1',
          'Prediction for sample # 1',
          'Considering decision rule on feature Sex with value 0',
          'Considering decision rule on feature Pclass with value 2',
          'Considering decision rule on feature Parch with value 1',
          'Considering decision rule on feature SibSp with value 0',
          'Considering decision rule on feature Pclass with value 3',
          'Predicted label: 0']


class TreePredTest(StageTest):

    @dynamic_test
    def test(self):
        pr = TestedProgram()
        pr.start()
        if not pr.is_waiting_input():
            raise WrongAnswer("You program should input two paths to the files split by space")
        output = pr.execute("test/data_stage5_train.csv test/data_stage5_test.csv").strip()
        res = output.split("\n")
        res = [x.strip() for x in res]
        if len(res) != 13:
            raise WrongAnswer("Wrong number of log messages.\n"
                              "Your model should visit four internal nodes making prediction for the first sample\n"
                              "and five internal nodes making prediction for the second sample.\n"
                              "Do not forget to add log messages showing the number of sample and its predicted label.")
        for i in range(1, 13):
            if i == 6:
                continue
            if res[i] != ANSWER[i]:
                if i == 5:
                    raise WrongAnswer("Wrong prediction for the first sample."
                                      "\nCorrect message template: 'Predicted label: 0'.")
                elif i == 12:
                    raise WrongAnswer("Wrong prediction for the second sample."
                                      "\nCorrect message template: 'Predicted label: 0'.")
                else:
                    raise WrongAnswer(f"Wrong log message on line {i+1}."
                                      "\nIt may be useful to check the traversal order in the recursive split function: go to the left child first."
                                      "\nCorrect message template: 'Considering decision rule on feature Sex with value 1'.")
        return CheckResult.correct()


if __name__ == '__main__':
    TreePredTest().run_tests()
