import argparse
from extract_one_recipe import get_recipe
from utils import print_json, profile


@profile
def main_for_one_link():
    parser = argparse.ArgumentParser(description='Print the recipe json from given link')
    parser.add_argument('link')
    args = parser.parse_args()

    dict_file = get_recipe(args.link)
    # TODO clean-preprocess data from link
    # TODO load pre-trained model
    # TODO predict on new set
    print_json(str(args.link).strip(), dict_file)
    # TODO print to file/console info about prediction


if __name__ == '__main__':
    main_for_one_link()
    # link = 'https://www.loveandlemons.com/focaccia/'
    # TODO add profile
    # TODO add logging
