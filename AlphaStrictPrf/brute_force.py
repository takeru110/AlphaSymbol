from itertools import combinations
from typing import List

from strict_prf import C, Expr, P, R, S, Z

MAX_P_ARITY = 2
EXPR_DEPTH = 2
MAX_C_ARGS = 2


# Step 1: Generate all possible PRF expressions (up to a certain depth)
def generate_expressions(
    depth: int,
    max_p_arity: int,
    max_c_args: int,
) -> List[Expr]:
    """
    Generate all possible PRF expressions up to a given depth.

    Args:
    - depth (int): The depth of recursion for generating expressions.
    - max_c_args (int): Maximum number of arguments allowed for C.

    Returns:
    - List[Expr]: List of generated PRF expressions.
    """
    # Base case: depth == 1, return the simplest expressions (Z, S, P)
    expressions = [Z(), S()]  # Base zero and successor functions

    # Add projection functions for the input arity (in this case, 6 inputs)
    for n in range(1, max_p_arity + 1):
        for i in range(1, n + 1):
            expressions.append(P(n, i))

    # If depth is more than 1, we can create more complex expressions
    if depth > 1:
        smaller_expressions = generate_expressions(
            depth - 1, max_p_arity, max_c_args
        )

        # Generate composite functions C with different number of arguments
        for num_args in range(1, max_c_args + 1):
            for args in combinations(smaller_expressions, num_args):
                for func in smaller_expressions:
                    expressions.append(C(func, *args))

        # Generate recursive functions R(Expr1, Expr2)
        for expr1 in smaller_expressions:
            for expr2 in smaller_expressions:
                expressions.append(R(expr1, expr2))

    return list(set(expressions))


# Step 2: Filter the list of expressions to keep only those that are semantically valid
def filter_valid_expressions(expressions: List[Expr]):
    valid_expressions = []
    for expr in expressions:
        if expr.validate_semantic():
            valid_expressions.append(expr)
    return valid_expressions


# Function to evaluate an expression on a given set of inputs
def evaluate_expression_on_input(expr: Expr, input_values: List[int]):
    try:
        return [expr.evaluate(*[x]) for x in input_values]
    except:
        return None  # Return None if the evaluation fails (due to wrong arity or errors)


# Step 3: Filter expressions based on the test case (input/output)
def filter_by_test_case(
    expressions: List[Expr], input_values: List[int], expected_output: List[int]
):
    matching_expressions = []
    for expr in expressions:
        result = evaluate_expression_on_input(expr, input_values)
        if result == expected_output:
            matching_expressions.append(expr)
    return matching_expressions


# Target inputs and outputs (provided in the problem)
target_input = [0, 1, 2, 3, 4, 5, 6]
target_output = [1, 0, 0, 0, 0, 0, 0]


# Step 1: Generate all possible PRF expressions
all_possible_expressions = generate_expressions(
    depth=EXPR_DEPTH, max_p_arity=MAX_P_ARITY, max_c_args=MAX_C_ARGS
)
print(f"Generated {len(all_possible_expressions)} expressions.")
with open("output/possible.txt", "w") as file:
    for item in all_possible_expressions:
        file.write(f"{item.parenthesized_string()}\n")

# Step 2: Filter out invalid expressions according to validate_semantic()
valid_expressions = filter_valid_expressions(all_possible_expressions)
print(f"Filtered down to {len(valid_expressions)} valid expressions.")
with open("output/valid.txt", "w") as file:
    for item in valid_expressions:
        file.write(f"{item.parenthesized_string()}\n")

# Step 3: Filter expressions that match the input/output test case
matching_expressions = filter_by_test_case(
    valid_expressions, target_input, target_output
)
print(f"Found {len(matching_expressions)} matching expressions.")
with open("output/matching.txt", "w") as file:
    for item in matching_expressions:
        file.write(f"{item.parenthesized_string()}\n")

# Output the matching expressions
for expr in matching_expressions:
    print(expr.parenthesized_string())
