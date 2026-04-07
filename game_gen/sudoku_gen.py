import random
import math
import hashlib
import json
import logging
import pandas as pd
from pathlib import Path

class ProgressiveSudokuGenerator:
    def __init__(self, sqrt_n):
        if not isinstance(sqrt_n, int) or sqrt_n <= 0:
            raise ValueError("sqrt(n) must be a positive integer.")

        self.sqrt_n = sqrt_n
        self.n = sqrt_n * sqrt_n

        if int(math.sqrt(self.n)) != sqrt_n:
            raise ValueError(f"sqrt(n)={sqrt_n} cannot generate a valid Sudoku grid.")

    def is_valid(self, board, row, col, num):
        for x in range(self.n):
            if board[row][x] == num or board[x][col] == num:
                return False

        start_row = (row // self.sqrt_n) * self.sqrt_n
        start_col = (col // self.sqrt_n) * self.sqrt_n

        for i in range(self.sqrt_n):
            for j in range(self.sqrt_n):
                if board[start_row+i][start_col+j] == num:
                    return False

        return True

    def solve(self, board):
        for row in range(self.n):
            for col in range(self.n):
                if board[row][col] == 0:
                    nums = list(range(1, self.n+1))
                    random.shuffle(nums)

                    for num in nums:
                        if self.is_valid(board, row, col, num):
                            board[row][col] = num
                            if self.solve(board):
                                return True
                            board[row][col] = 0
                    return False
        return True

    def generate_solution(self):
        board = [[0]*self.n for _ in range(self.n)]
        self.solve(board)
        return board

    def count_solutions(self, board, limit=2):
      board = [row[:] for row in board]
      count = 0

      def backtrack():
          nonlocal count

          if count >= limit:
              return

          for r in range(self.n):
              for c in range(self.n):
                  if board[r][c] == 0:
                      for num in range(1, self.n+1):
                          if self.is_valid(board, r, c, num):
                              board[r][c] = num
                              backtrack()
                              board[r][c] = 0
                      return

          count += 1

      backtrack()
      return count

    def remove_cells(self, solution, remove_count):
      puzzle = [row[:] for row in solution]

      cells = [(r,c) for r in range(self.n) for c in range(self.n)]
      random.shuffle(cells)

      removed_cells = cells[:remove_count]

      for r,c in removed_cells:
          puzzle[r][c] = 0

      remaining_removed = removed_cells[:]   # blanks we can refill later

      return puzzle, remaining_removed

    def fill_cells(self, puzzle_board, solution, available_cells, fill_count, verbose=False):
        puzzle = [row[:] for row in puzzle_board]
        cells = available_cells[:]
        random.shuffle(cells)

        used = cells[:fill_count]
        remaining = cells[fill_count:]

        for r, c in used:
          if verbose >= 2:
            print(f'filling cell {r}, {c} with {solution[r][c]}')
          puzzle[r][c] = solution[r][c]

        if verbose:
          print('Available cells for board = ')
          print(available_cells)

          print("Used cells = ")
          print(used)

          print("Puzzle board old : \n")
          print_board(puzzle_board)

          print("Puzzle board new : \n")
          print_board(puzzle)

        # for i in range(min(fill_count, len(cells))):
        #     r,c = cells[i]
        #     puzzle[r][c] = solution[r][c]

        return puzzle, remaining


def print_board(board):
    for row in board:
        print(" ".join(str(x) if x!=0 else "." for x in row))



def board_hash(board):
    flat = ''.join(str(x) for row in board for x in row)

    return hashlib.sha1(flat.encode()).hexdigest()[:16]

def make_board_id(h, size, difficulty):

    return f"S{size}_{difficulty}_{h}"


def solution_hash(board):
    return ''.join(str(x) for row in board for x in row)


def difficulty_removals(n):

    total = n*n

    return {
        "hard": int(total * 0.70),
        "medium": int(total * 0.55),
        "easy": int(total * 0.40)
    }


def sudoku_interface(sqrt_n, difficulty):

    generator = ProgressiveSudokuGenerator(sqrt_n)
    solution = generator.generate_solution()

    total = generator.n * generator.n

    # removal ratios
    remove_map = {
        "hard": int(total * 0.70),
        "medium": int(total * 0.50),
        "easy": int(total * 0.40)
    }

    print(remove_map["hard"], remove_map["easy"])
    if difficulty == "all":

        hard_remove = remove_map["hard"]
        hard_board, remaining_cells = generator.remove_cells(solution, hard_remove)


        medium_fill = hard_remove - remove_map["medium"]
        medium_board, remaining_cells = generator.fill_cells(hard_board, solution, remaining_cells, medium_fill)

        easy_fill = remove_map["medium"] - remove_map["easy"]
        easy_board, remaining_cells = generator.fill_cells(medium_board, solution, remaining_cells, easy_fill)

        boards = {
            "hard": hard_board,
            "medium": medium_board,
            "easy": easy_board
        }

        for diff in ["hard","medium","easy"]:
            print(f"\n{diff.upper()} BOARD\n")
            print_board(boards[diff])

        print("\nSOLUTION\n")
        print_board(solution)

        return boards, solution, generator

    else:

        remove_count = remove_map[difficulty]
        puzzle,_ = generator.remove_cells(solution, remove_count)

        print("\nSTARTER BOARD\n")
        print_board(puzzle)

        print("\nSOLUTION\n")
        print_board(solution)

        return puzzle, solution, generator


def generate_progressive_puzzles(gen, solution):

    remove_map = difficulty_removals(gen.n)

    hard_remove = remove_map["hard"]

    while True:

        hard_board, removed_cells = gen.remove_cells(solution, hard_remove)

        if gen.count_solutions(hard_board) == 1:
            break

    medium_fill = remove_map["hard"] - remove_map["medium"]

    medium_board, removed_cells = gen.fill_cells(
        hard_board,
        solution,
        removed_cells,
        medium_fill
    )

    easy_fill = remove_map["medium"] - remove_map["easy"]

    easy_board, removed_cells = gen.fill_cells(
        medium_board,
        solution,
        removed_cells,
        easy_fill
    )

    return {
        "hard": hard_board,
        "medium": medium_board,
        "easy": easy_board
    }, remove_map



def generate_dataset_2(sqrt_n, target_boards):

    gen = ProgressiveSudokuGenerator(sqrt_n)

    seen_solutions = set()
    seen_root = set()
    dataset = []
    tries = 0


    while len(dataset) < target_boards:

        solution = gen.generate_solution()

        shash = solution_hash(solution)

        if shash in seen_solutions:
            tries += 1
            if tries > target_boards//2:
                break
            continue

        seen_solutions.add(shash)

        puzzles, remove_map = generate_progressive_puzzles(gen, solution)

        root_hash = board_hash(puzzles['hard'])

        if root_hash in seen_root:
            continue
        seen_root.add(root_hash)

        for diff, puzzle in puzzles.items():

            pid = make_board_id(root_hash, gen.n, diff)

            dataset.append({
                "board_id": pid,
                "root_hash": root_hash,
                "solution_id": shash,
                "size": gen.n,
                "difficulty": diff,
                "missing_cells": remove_map[diff],
                "puzzle_grid": json.dumps(puzzle),
                "solution_grid": json.dumps(solution)
            })



    return dataset

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # data = generate_dataset_2(sqrt_n=2, target_boards=10000)
    data = generate_dataset_2(sqrt_n=4, target_boards=1000)
    df = pd.DataFrame(data)
    # save_path = Path(__file__).resolve() / '../../data' / 'four_grid.csv'
    save_path = Path(__file__).resolve() / '../../data' / 'sixteen_grid.csv'

    df.to_csv(str(save_path.resolve()))