import logging

import pandas as pd
import tensorflow
from config import BATCH_SIZE, THRESHOLD, LOG_FILE

# log-file will be created in the main dir
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')



def eval_on_one_page(sent2vec_one_page, X_meta_one_page, y_one_page, model, text):
    """Load model from file and evaluate on data from one page
    :param sent2vec_one_page : ndArray,
    :param X_meta_one_page: ndArray,
    :param y_one_page: ndArray,
    :param model:str - path to the model
    :param text : ndArray[str]
    """

    new_model1 = tensorflow.keras.models.load_model(model)
    preds = new_model1.predict([sent2vec_one_page, X_meta_one_page])
    pred_df = pd.DataFrame(columns=['text', 'pred_label'])
    pred_df['text'] = text

    pred_df['proba'] = preds
    pred_df['label'] = y_one_page
    pred_df['label'] = pred_df['label'].apply(lambda x: 'Instructions' if x == 0 else 'Recepie')
    pred_df['pred_label'] = pred_df['proba'].apply(lambda x: 'Recepie' if x > THRESHOLD else 'Instructions')
    logging.info(f'{model}')
    logging.info(pred_df[['label', 'pred_label','proba']])

    score1 = new_model1.evaluate([sent2vec_one_page, X_meta_one_page], y_one_page,
                                 batch_size=BATCH_SIZE, verbose=1)

    logging.info(f'{model} Loss score : {round(score1[0], 2)}')
    logging.info(f'{model} Accuracy/recall Evaluation : {round(score1[1], 2)}')
