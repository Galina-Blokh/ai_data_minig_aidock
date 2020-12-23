import argparse
from extract_one_recipe import get_recipe

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print the recipe json from given link')
    parser.add_argument('link')
    args = parser.parse_args()

    get_recipe(args.link)
    # get_recipe('https://www.loveandlemons.com/mashed-cauliflower-recipe/')
