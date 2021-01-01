import logging

# for scraping part
BATCHES = 20
LINK_PATTERN = 'https://www.loveandlemons'
URL = 'https://www.loveandlemons.com/recipes/'
LOG_FILE = 'recipes_logging.log'
FILE_LINKS_NAME = 'all_recipes_links.txt'
EMPTY_LINKS = 'no_recipe_page.txt'
DATA_FILE = 'recipes.pkl'

# constants  for preprocess  data
DIGIT_RX = "(\d+([\.|,]\d+)?[\w]*[\s|-])|[^A-Za-z\,()\.'\-: ]{1,7}"
SYMBOL_RX = "[/(/)\-/*/,]|[^ -~]"
DOT_RX = "\.{1,4}|\:"

# constants for model training
TEST_SIZE = 0.2
BATCH_SIZE = 256  # 64 loss: 0.2981 - accuracy: 0.9498 #256 --> acc0.95 loss0.24, 128-->acc94 loss 0.26 with dropout 0.5
EPOCHS = 200
THRESHOLD = 0.5
EMBEDDING_DIM = 128
TRAIN_DATA_CLEAN = 'data/train_data_clean.pkl'
TEST_DATA_CLEAN = 'data/test_data_clean.pkl'
MODEL_NAME = 'data/lstm_concat.h5'

MAX_SEQ_LEN = 121
VOCAB_SIZE = 2200

# # log-file will be created in the same dir
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
