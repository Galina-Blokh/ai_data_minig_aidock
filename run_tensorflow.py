import logging

import pandas as pd
import tensorflow
from config import BATCH_SIZE, THRESHOLD, LOG_FILE
from utils import profile

# log-file will be created in the main dir
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')



