import numpy as np
import timeit
from src.ConnectFour import ConnectFour


def run_test(test, index, cnct4, test_method, do_print=True):
  cnct4.board = test["board"] #np.array(test["board"], dtype=np.byte) 
  cnct4.level = np.array(test["level"]) if ("level" in test and test["level"] is not None) else get_level(cnct4)
  cnct4.last_col = test["col"]
  cnct4.player = -test["player"]
  cnct4.num_moves = max(7, 42 - 7 - sum(test["level"]))
  try:
    actual_result = test_method()
    assert actual_result == test['result']
  except AssertionError:
    if (do_print):
      print(f"Test {index} failed ({test['name']}): should have been {test['result']} but was {actual_result}")
      print(cnct4.board)
      print(cnct4.level)
    i = 0
    return 1
  return 0

def run_tests(tests, cnct4, test_method, do_print=True):
  num_errs = 0
  for i, test in enumerate(tests):
    num_errs += run_test(test, i, cnct4, test_method, do_print)
  if (do_print):
    print(f"Passed {len(tests) - num_errs} of {len(tests)} tests")

game_over_tests = [
         {"result": (True, 1), "col": 2, "player": 1, "board": [
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0]],
          "level":[5, 5, 1, 2, 3, 4, 5],
          "name": "win diagonal down - start from edge"
         },
         {"result": (True, 1), "col": 3, "player": 1, "board": [
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0]],
          "level":[5, 5, 1, 2, 3, 4, 5],
          "name": "win diagonal down - start from center"
         },
        {"result": (True, 1), "col": 2, "player": 1, "board": [
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0]],
          "level":[5, 5, 1, 5, 5, 5, 5],
          "name": "win vertical"
         },
         {"result": (True, 1), "col": 4, "player": 1, "board": [
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1, 0, 0]],
          "level":[5, 4, 4, 4, 4, 5, 5],
          "name": "win horizontal - start from edge"
         },
         {"result": (True, 1), "col": 2, "player": 1, "board": [
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1, 0, 0]],
          "level":[5, 4, 4, 4, 4, 5, 5],
          "name": "win horizontal - start from center"
         },
         {"result": (True, 1), "col": 5, "player": 1, "board": [
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]],
          "level":[5, 5, 3, 2, 1, 0, 5],
          "name": "win diagonal up - start from edge"
         },
         {"result": (True, 1), "col": 4, "player": 1, "board": [
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]],
          "level":[5, 5, 3, 2, 1, 0, 5],
          "name": "win diagonal up - start from center"
         },{"result": (False, 0), "col": 2, "player": 1, "board": [
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0]],
          "level":[5, 5, 1, 2, 5, 4, 5],
          "name": "no win diagonal down - missing center"
         },
        {"result": (False, 0), "col": 2, "player": 1, "board": [
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0],
                  [0, 0,-1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0]],
          "level":[5, 5, 1, 5, 5, 5, 5],
          "name": "no win vertical - missing center"
         },
         {"result": (False, 0), "col": 4, "player": 1, "board": [
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 1, 1, 0, 0]],
          "level":[5, 4, 5, 4, 4, 5, 5],
          "name": "no win horizontal - missing center"
         },
         {"result": (False, 0), "col": 4, "player": 1, "board": [
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]],
          "level":[5, 5, 3, 2, 1, 0, 5],
          "name": "no win diagonal up - missing center"
         },
         {"result": (False, 0), "col": 2, "player": 1, "board": [
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]],
          "level":[5, 5, 1, 2, 3, 5, 5],
          "name": "no win diagonal down - missing edge"
         },
        {"result": (False, 0), "col": 2, "player": 1, "board": [
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0],
                  [0, 0,-1, 0, 0, 0, 0]],
          "level":[5, 5, 1, 5, 5, 5, 5],
          "name": "no win vertical - missing edge"
         },
         {"result": (False, 0), "col": 4, "player": 1, "board": [
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 0, 0]],
          "level":[5, 4, 4, 4, 4, 5, 5],
          "name": "no win horizontal - missing edge"
         },
         {"result": (False, 0), "col": 4, "player": 1, "board": [
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0,-1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]],
          "level":[5, 5, 3, 2, 1, 0, 5],
          "name": "no win diagonal up - missing edge"
         },
         {"result": (True, 1), "col": 6, "player": 1, "board": [
                  [0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]],
          "level":[-1,-1,-1,-1,-1,-1,-1],
          "name": "win on last move"
         },
         {"result": (True, 0), "col": 6, "player": 1, "board": [
                  [0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, -1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]],
          "level":[-1,-1,-1,-1,-1,-1,-1],
          "name": "tie - board full"
         },
]

cnct4 = ConnectFour()
#test_method = lambda: cnct4.game_over_optimized()
test_method = lambda: cnct4.game_over()
#test_method2 = lambda: cnct4.game_over_conv()
expanded_game_over_tests = game_over_tests[:]
expanded_game_over_tests += [{"result": (test["result"][0], -test["result"][1]), "board": np.array(test["board"])*-1, "name": test["name"] + " negated", "col": test["col"], "player": -1*test["player"], "level": test["level"] if "level" in test else None} for test in game_over_tests]
for item in expanded_game_over_tests:
  item["board"] = np.array(item["board"])
run_tests(expanded_game_over_tests, cnct4, test_method)
t = timeit.timeit(lambda: run_tests(expanded_game_over_tests, cnct4, test_method, do_print=False), number=5000)
print(t)
#t = timeit.timeit(lambda: run_tests(expanded_game_over_tests, cnct4, test_method2, do_print=False), number=1000)
#print(t)

