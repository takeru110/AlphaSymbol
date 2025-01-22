import argparse
import logging
import random

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str)
    parser.add_argument("-o_train", "--output_file_train", type=str)
    parser.add_argument("-o_test", "--output_file_test", type=str)
    parser.add_argument(
        "-r",
        "--ratio",
        type=float,
        default=0.1,
        help="The ratio of the test data",
    )
    args = parser.parse_args()

    # Read all lines from the input file
    with open(args.input_file, "r") as file:
        lines = file.readlines()

    n_sample = len(lines)
    dividor = int(n_sample * args.ratio)

    # Shuffle the lines randomly
    header = lines[0]
    lines = lines[1:]

    logging.info("Shuffing data!")
    random.shuffle(lines)

    # Split into two parts
    lines_test = lines[:dividor]
    lines_train = lines[dividor:]

    # Write the 200k lines to one file
    with open(args.output_file_test, "w") as file:
        file.write(header)
        file.writelines(lines_test)

    # Write the 10k lines to another file
    with open(args.output_file_train, "w") as file:
        file.write(header)
        file.writelines(lines_train)

    logging.info("File saved!")
