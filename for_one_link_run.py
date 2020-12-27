import argparse
from extract_one_recipe import get_recipe
from utils import print_json, timeit


@timeit
def main_for_one_link():
    parser = argparse.ArgumentParser(description='Print the recipe json from given link')
    parser.add_argument('link')
    args = parser.parse_args()

    g = get_recipe(args.link)
    print_json(args.link, g)


if __name__ == '__main__':
    main_for_one_link()
    # link = 'https://www.loveandlemons.com/focaccia/'
    #

# TODO here will be run a prediction and print for test set/link
