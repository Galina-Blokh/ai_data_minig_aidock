# for scraping part
BATCHES = 20
LINK_PATTERN = 'https://www.loveandlemons'
URL = 'https://www.loveandlemons.com/recipes/'
LOG_FILE = 'recipes_logging.log'
FILE_LINKS_NAME = 'all_recipes_links.txt'
EMPTY_LINKS = 'no_recipe_page.txt'
TEST_LINKS_FILE = 'test_links.txt'
DATA_FILE = 'new_recipe.pkl'

# constants  for preprocess  data
DIGIT_RX = "(\d+([\.|,]\d+)?[\w]*[\s|-])|[^A-Za-z\,()\.'\-: ]{1,7}"
SYMBOL_RX = "[/(/)\-/*/,]|[^ -~]"
DOT_RX = "\.{1,4}|\:"

# for model , preprocess, training
TEST_SIZE = 0.2
BATCH_SIZE = 128
EPOCHS = 200
THRESHOLD = 0.5
EMBEDDING_DIM = 128
TRAIN_DATA_CLEAN = '/data/new_train_data_clean.pkl'
TEST_DATA_CLEAN = '/data/new_test_data_clean.pkl'
MODEL_NAME = '/data/new_my_model.h5'
MAX_SEQ_LEN = 121
VOCAB_SIZE = 2284  # it is changed from 2263. It depends on scrapping part -
# links to scrap each time goes in different order --> when you take out first 10 links from
# all_links.txt it will be 10 different links each scraper run --> the vocabulary every time will be different size
