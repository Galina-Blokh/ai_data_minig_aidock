# for scraping part
BATCHES = 20
LINK_PATTERN = 'https://www.loveandlemons'
URL = 'https://www.loveandlemons.com/recipes/'
LOG_FILE = 'recipes_logging.log'
TEST_LINKS_FILE = 'test_links.txt'
DATA_FILE = 'new_recipe.pkl'

# for model , preprocess, training
TEST_SIZE = 0.2
BATCH_SIZE = 16
EPOCHS = 5
# THRESHOLD = 0.5
EMBEDDING_DIM = 32
TRAIN_DATA_CLEAN = '/data/train_data_clean.pkl'
TEST_DATA_CLEAN = '/data/test_data_clean.pkl'
MODEL_NAME = '/data/distilbert.h5'
MAX_SEQ_LEN = 217

