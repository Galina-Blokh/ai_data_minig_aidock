import argparse
from extract_one_recipe import get_recipe
from utils import print_json, timeit


@timeit
def main():
    parser = argparse.ArgumentParser(description='Print the recipe json from given link')
    parser.add_argument('link')
    args = parser.parse_args()

    g = get_recipe(args.link)

    print_json(str(args.link).strip(), g)


if __name__ == '__main__':
    main()

# TODO here will be run a prediction and print for test set/link
