import csv
import argparse
from pathlib import Path


def generate_constant(n: int) -> str:
    """
    Generate a constant string with n levels of C() nesting.
    For n=0, returns "Z()"
    For n=1, returns "C(S(), Z())"
    For n=2, returns "C(C(S(), Z()), S())"
    And so on...
    """
    if n == 0:
        return "Z()"
    
    # For n > 0, we build the pattern recursively
    inner = generate_constant(n - 1)
    return f"C({inner}, S())"


def generate_constants_file(output_path: Path, num_constants: int = 20):
    """
    Generate a CSV file containing constants with increasing levels of nesting.
    
    Args:
        output_path: Path to the output CSV file
        num_constants: Number of constants to generate (default: 20)
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        for i in range(num_constants):
            constant = generate_constant(i)
            writer.writerow([constant])


def parse_args():
    parser = argparse.ArgumentParser(description='Generate constants with increasing levels of nesting')
    parser.add_argument('-o', '--output',
                      default='./output.csv',
                      help='Output file path (default: ./data/prfndim/constants10.csv)')
    parser.add_argument('-n', '--number',
                      type=int,
                      default=20,
                      help='Number of constants to generate (default: 20)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_constants_file(output_path, args.number)
    print(f"Generated {output_path} with {args.number} constants")
