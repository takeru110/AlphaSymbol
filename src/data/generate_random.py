import argparse
import logging
import pandas as pd
import random
from pathlib import Path
from typing import Optional, Dict, List
import linecache
import csv
import os
import numpy as np
import numpy.typing as npt

from prfndim.prfndim import C, Expr, OverflowError, P, R, S, Z
from src.data.prfndim_utils import expr_eval_safe

BATCH_SIZE = 200

class InputExprs:
    def __init__(self, input_file: Path, max_arity: int, max_exprs_in_memory: int = 100000, pop_counter_threshold: int = 1000):
        self.input_file = input_file
        self.max_arity = max_arity
        self.max_exprs_in_memory = max_exprs_in_memory
        self.pop_counter_threshold = pop_counter_threshold
        # Initialize as nested list where index represents arity
        self.exprs_by_arity: List[List[Expr]] = [[] for _ in range(max_arity + 1)]
        self.pop_counter = 0
        
        # Read header to get the index of 'expr' column
        with open(self.input_file, 'r') as f:
            header = next(csv.reader(f))
            try:
                self.expr_column_index = header.index('expr')
            except ValueError:
                raise ValueError("CSV file must contain a column named 'expr'")
        
        # Get total number of lines in file (excluding header)
        with open(self.input_file, 'r') as f:
            # Skip header
            next(f)
            self.total_lines = sum(1 for _ in f)
        
        self._load_random_exprs()

    def _get_random_line_numbers(self) -> List[int]:
        """Get random line numbers to read from the file."""
        if self.total_lines <= self.max_exprs_in_memory:
            # If file is smaller than max_exprs_in_memory, use all lines
            return list(range(2, self.total_lines + 2))  # +2 because of 1-based line numbers and header
        else:
            # Randomly select max_exprs_in_memory lines
            # +2 because of 1-based line numbers and header
            return sorted(random.sample(range(2, self.total_lines + 2), self.max_exprs_in_memory))

    def _load_random_exprs(self):
        """Load random expressions from input file into memory."""
        # Skip reloading if we already have expressions for each arity
        if self.total_lines <= self.max_exprs_in_memory:
            # Check if we have at least one expression exprs_by_arity have been already set.
            if all(len(exprs) > 0 for exprs in self.exprs_by_arity):
                return
        
        # Clear existing expressions
        for arity in range(self.max_arity + 1):
            self.exprs_by_arity[arity] = []
        
        # Get random line numbers to read
        line_numbers = self._get_random_line_numbers()
        
        # Read selected lines using linecache
        for line_num in line_numbers:
            line = linecache.getline(str(self.input_file), line_num).strip()
            if not line:  # Skip empty lines
                continue
                
            # Parse CSV line
            try:
                row = next(csv.reader([line]))
                if len(row) <= self.expr_column_index:
                    continue
                expr_str = row[self.expr_column_index]
                expr = expr_eval_safe(expr_str)
                
                if expr.arity is None:
                    self.exprs_by_arity[0].append(expr)
                elif expr.arity <= self.max_arity:
                    self.exprs_by_arity[expr.arity].append(expr)
            except (IndexError, StopIteration, OverflowError):
                continue
        
        # Clear linecache to free memory
        linecache.clearcache()

    def get_random_expr(self, arity: int) -> Optional[Expr]:
        """Get a random expression of specified arity."""
        if arity < 0 or arity > self.max_arity or not self.exprs_by_arity[arity]:
            return None
            
        self.pop_counter += 1
        if self.pop_counter >= self.pop_counter_threshold:
            self._load_random_exprs()
            self.pop_counter = 0
            
        return random.choice(self.exprs_by_arity[arity])

OUTPUT_FILE = None
data_buffer: list[Expr] = []
saved_expr_counter = 0


def add_data(new_data: Expr):
    global data_buffer

    # データをバッファに追加
    data_buffer.append(new_data)

    # バッチサイズを超えた場合、CSVファイルに追記してバッファをクリア
    if len(data_buffer) >= BATCH_SIZE:
        save_to_csv(data_buffer)
        data_buffer.clear()


def save_to_csv(data):
    global saved_expr_counter
    # データをDataFrameに変換
    df = pd.DataFrame()
    df["expr"] = data

    saved_expr_counter += len(data)
    logging.info(
        "Output file is updated. There are %d exprs", saved_expr_counter
    )

    # ファイルが存在しない場合は新規作成、存在する場合は追記
    assert OUTPUT_FILE.exists(), "Output file does not exist"
    df.to_csv(OUTPUT_FILE, mode="a", index=False, header=False)


def output_bytes_not_const(expr: Expr, eq_domain: list[npt.NDArray]) -> bytes:
    assert expr.arity is not None
    ans: npt.NDArray = np.array([expr.eval(*x) for x in eq_domain[expr.arity]])
    return ans.tobytes()


def output_bytes_const(
    expr: Expr, input_size: int, eq_domain: list[npt.NDArray]
) -> bytes:
    assert input_size >= 1
    ans = np.array([expr.eval(0) for _ in range(len(eq_domain[input_size]))])
    return ans.tobytes()


def init_exprs_from_csv(
    init_file: Path, max_arity: int, eq_domain: list[npt.NDArray]
) -> list[set[bytes]]:
    """Initialize outputs from CSV file.
    
    Args:
        init_file: Path to the CSV file containing expressions
        max_arity: Maximum arity of expressions to consider
        eq_domain: List of numpy arrays for each arity, used for evaluation
        
    Returns:
        List of sets of bytes, where each set contains the outputs of expressions for that arity
    """
    outputs: list[set[bytes]] = [set() for _ in range(max_arity + 1)]
    
    with open(init_file, 'r') as f:
        reader = csv.reader(f)
        # Read header to get the index of 'expr' column
        header = next(reader)
        try:
            expr_column_index = header.index('expr')
        except ValueError:
            raise ValueError("CSV file must contain a column named 'expr'")
        
        # Process each line
        for row in reader:
            if len(row) <= expr_column_index:
                continue
                
            try:
                expr: Expr = expr_eval_safe(row[expr_column_index])
                if expr.arity is None:
                    try:
                        for input_size in range(1, max_arity + 1):
                            outputs[input_size].add(
                                output_bytes_const(expr, input_size, eq_domain)
                            )
                    except OverflowError:
                        continue
                else:
                    try:
                        outputs[expr.arity].add(output_bytes_not_const(expr, eq_domain))
                    except OverflowError:
                        continue
            except (IndexError, OverflowError):
                continue
    
    return outputs


def if_not_visited_then_update_const(
    exprs: list[list[Expr]],
    outputs: list[set[bytes]],
    expr: Expr,
    max_dim: int,
    eq_domain: list[npt.NDArray],
) -> tuple[list[set[bytes]], bool]:
    assert expr.arity is None
    is_visited = True
    for dim in range(1, max_dim + 1):
        try:
            # when OverflowError occurs, it occurs in all iteration
            out_bytes = output_bytes_const(expr, dim, eq_domain)
        except OverflowError:
            return outputs, False
        if out_bytes in outputs[dim]:
            continue
        outputs[dim].add(out_bytes)
        is_visited = False
    is_new = not is_visited
    return outputs, is_new


def if_not_visited_then_update_not_const(
    exprs: list[list[Expr]],
    outputs: list[set[bytes]],
    expr: Expr,
    eq_domain: list[npt.NDArray],
) -> tuple[list[set[bytes]], bool]:
    assert expr.arity is not None
    try:
        out_bytes = output_bytes_not_const(expr, eq_domain)
    except OverflowError:
        return outputs, False
    if out_bytes in outputs[expr.arity]:
        return outputs, False
    outputs[expr.arity].add(out_bytes)
    return outputs, True


def generate_random(
    sample_num,
    max_arity,
    max_c_args,
    max_r_args,
    eq_domain,
    output_path: Path,
    init_csv: Path,
):
    global OUTPUT_FILE
    if output_path.exists():
        output_path.unlink()
    output_path.touch()
    OUTPUT_FILE = output_path
    with open(OUTPUT_FILE, "w") as f:
        f.write("expr\n")

    # Initialize outputs from CSV and create InputExprs
    outputs = init_exprs_from_csv(init_csv, max_arity, eq_domain)
    input_exprs = InputExprs(init_csv, max_arity)
    new_exprs: list[Expr] = []

    while len(new_exprs) < sample_num:
        logging.debug(f"Current number of exprs: {len(new_exprs)}")
        if random.random() < 0.5:
            # create new expression with C
            base_arity: int = random.randint(1, min(max_arity, max_c_args - 1))
            arg_arity: int = random.randint(1, max_arity)
            
            base = input_exprs.get_random_expr(base_arity)
            if base is None:
                continue
                
            args = []
            for _ in range(base_arity):
                arg = input_exprs.get_random_expr(arg_arity)
                if arg is None:
                    break
                args.append(arg)
            else:  # Only execute if the for loop completed without break
                args = tuple(args)
                new_expr_c = C(base, *args)

                # Check new expression is already visited
                if new_expr_c.arity is None:
                    outputs, is_new = if_not_visited_then_update_const(
                        input_exprs,
                        outputs,
                        new_expr_c,
                        max_arity,
                        eq_domain,
                    )
                    if is_new:
                        new_exprs.append(new_expr_c)
                        add_data(new_expr_c)
                else:
                    outputs, is_new = if_not_visited_then_update_not_const(
                        input_exprs, outputs, new_expr_c, eq_domain
                    )
                    if is_new:
                        new_exprs.append(new_expr_c)
                        add_data(new_expr_c)
        else:
            # Create new expression with R
            while True:
                try:
                    term_arity = random.randint(1, (max_r_args - 1) // 2)
                    step_arity = random.randint(
                        term_arity + 1, min(term_arity + 1, max_arity)
                    )
                    base_arity = random.randint(0, step_arity - term_arity - 1)

                    term = input_exprs.get_random_expr(term_arity)
                    if term is None:
                        continue

                    steps = []
                    for _ in range(term_arity):
                        step = input_exprs.get_random_expr(step_arity)
                        if step is None:
                            break
                        steps.append(step)
                    else:  # Only execute if the for loop completed without break
                        steps = tuple(steps)
                        bases = []
                        for _ in range(term_arity):
                            base = input_exprs.get_random_expr(base_arity)
                            if base is None:
                                break
                            bases.append(base)
                        else:  # Only execute if the for loop completed without break
                            bases = tuple(bases)
                            new_expr_r = R(term, *steps, *bases)
                            
                            if new_expr_r.arity is None:
                                outputs, is_updated = if_not_visited_then_update_const(
                                    input_exprs,
                                    outputs,
                                    new_expr_r,
                                    max_arity,
                                    eq_domain,
                                )
                                if is_updated:
                                    new_exprs.append(new_expr_r)
                                    add_data(new_expr_r)
                            else:
                                outputs, is_updated = if_not_visited_then_update_not_const(
                                    input_exprs, outputs, new_expr_r, eq_domain
                                )
                                if is_updated:
                                    new_exprs.append(new_expr_r)
                                    add_data(new_expr_r)
                            break
                except ValueError:
                    continue

    logging.debug(f"Final number of exprs: {len(new_exprs)}")

    save_to_csv(data_buffer)
    data_buffer.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate expressions randomly"
    )

    parser.add_argument("-s", "--sample", type=int, required=True)
    parser.add_argument("-a", "--max_arity", type=int, required=True)
    parser.add_argument("-c", "--max_c_args", type=int, required=True)
    parser.add_argument("-r", "--max_r_args", type=int, required=True)
    parser.add_argument("--sample_num", type=int, default=10)
    parser.add_argument("--sample_max", type=int, default=10)
    parser.add_argument("-o", "--output", type=str, default="output.csv")
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--init_csv", type=str, required=True)

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s-%(levelname)s-%(message)s",
        force=True,
    )

    eq_domain = [np.zeros((1))] + [
        np.random.randint(1, args.sample_max + 1, size=(args.sample_num, dim))
        for dim in range(1, args.sample_max + 1)
    ]
    eq_domain[1] = np.array(range(10)).reshape(10, 1)

    output_path = None if args.output is None else Path(args.output)
    init_csv_path = Path(args.init_csv)

    exprs = generate_random(
        args.sample,
        args.max_arity,
        args.max_c_args,
        args.max_r_args,
        eq_domain,
        output_path,
        init_csv_path,
    )
