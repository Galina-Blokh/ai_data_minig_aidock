import logging
import os
import tensorflow
import matplotlib.pyplot as plt
import pandas as pd
from config import BATCH_SIZE, EPOCHS, MODEL_NAME, TRAIN_DATA_CLEAN, TEST_DATA_CLEAN, LOG_FILE
from preprocess import get_model, tfidf

# log-file will be created in the main dir
from utils import profile

logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# @profile
def model_train(train_data_clean_path=TRAIN_DATA_CLEAN, test_data_clean_path=TEST_DATA_CLEAN):
    """Read train and test data from pkl files
            count max len sent/sequence
            count vocabulary size
            transform data into sequences
            split data into nlp and meta sets for test and train
            train the model, evaluate and plot loss vs val_loss
            save the model into 'data/my_model.h5' file
    :param train_data_clean_path: str - default  config.TRAIN_DATA_CLEAN
    :param test_data_clean_path: str - default config.TEST_DATA_CLEAN
    :return void
    """

    train_data_clean = pd.read_pickle(os.getcwd() + train_data_clean_path)
    logging.info(f'In train {round(train_data_clean.label.value_counts() / len(train_data_clean) * 100, 2)}')

    test_data_clean = pd.read_pickle(os.getcwd() + test_data_clean_path)
    logging.info(f'In test {round(test_data_clean.label.value_counts() / len(test_data_clean) * 100, 2)}')

    # max len sequence count will be constanta at the end  (it is 121 in train - we will use it)
    max_sequence_length = train_data_clean['clean_paragraph_len'].max()
    logging.info(f'The max_sequence_len is {max_sequence_length}')
    # vocab_size - count in train set
    results = set()
    train_data_clean.remove_stop_words.str.split().apply(results.update)
    vocab_size = len(results)
    logging.info(f'The vocab_size is {vocab_size}')

    # sent to sequence only for  NLP TRAIN
    tf_idf_train = tfidf(train_data_clean.remove_stop_words, vocab_size)
    # for other features train
    X_meta_train = train_data_clean[['sent_count', 'num_count', 'clean_paragraph_len', 'verb_count', 'contains_pron']]
    y_train = train_data_clean['label']

    # for NLP TEST
    tf_idf_test = tfidf(test_data_clean.remove_stop_words, vocab_size)
    # for other features test
    X_meta_test = test_data_clean[['sent_count', 'num_count', 'clean_paragraph_len', 'verb_count', 'contains_pron']]
    y_test = test_data_clean['label']

    # create and train the MODEL
    concat_biLstm = get_model(tf_idf_train, X_meta_train, results)
    concat_biLstm.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=[tensorflow.keras.metrics.Recall(),
                                   tensorflow.keras.metrics.Precision(),
                                   'accuracy',
                                   tensorflow.keras.metrics.AUC()])
    es = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss')
    history = concat_biLstm.fit([tf_idf_train, X_meta_train],
                                y_train,
                                batch_size=BATCH_SIZE,
                                epochs=EPOCHS,
                                validation_split=0.2,
                                callbacks=[es])
    # evaluate the model
    score = concat_biLstm.evaluate([tf_idf_test, X_meta_test], y_test, batch_size=BATCH_SIZE, verbose=1)
    logging.info(f'Model Loss score: {round(score[0], 2)}')
    logging.info(f'Model Recall score: {round(score[1], 2)}')
    logging.info(f'Model Precision score: {round(score[2], 2)}')
    logging.info(f'Model Accuracy Evaluation : {round(score[3], 2)}')
    logging.info(f'Model AUC Evaluation : {round(score[4], 2)}')

    # plot the loss
    plt.figure(figsize=(15, 4))
    plt.plot(history.history['loss'], 'r', label=f'loss  ')
    plt.plot(history.history['val_loss'], 'g--', label=f'val_loss ')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('The Loss vs val_Loss LSTM Concat, activation = relu')
    plt.legend(loc='best')
    plt.show()

    concat_biLstm.save(MODEL_NAME)
    logging.info(f'The model is saved to {MODEL_NAME}')


if __name__ == '__main__':
    model_train()
