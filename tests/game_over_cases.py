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
