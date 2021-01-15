import logging
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Bidirectional, LSTM, Embedding, Dense, Dropout
from config import BATCH_SIZE, EPOCHS, MODEL_NAME, TRAIN_DATA_CLEAN, TEST_DATA_CLEAN, LOG_FILE, EMBEDDING_DIM, THRESHOLD
from preprocess import tfidf
# log-file will be created in the main dir
from utils import profile
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


@profile
def eval_on_one_page(tfidf_one_page, X_meta_one_page, y_one_page, model, text):
    """Load model from file and evaluate on data from one page
    :param tfidf_one_page : ndArray,
    :param X_meta_one_page: ndArray,
    :param y_one_page: ndArray,
    :param model:str - path to the model
    :param text : ndArray[str]
    """

    new_model1 = tf.keras.models.load_model(model)
    preds = new_model1.predict([tfidf_one_page, X_meta_one_page])
    pred_df = pd.DataFrame(columns=['text', 'pred_label'])
    pred_df['text'] = text

    pred_df['proba'] = preds
    pred_df['label'] = y_one_page
    pred_df['label'] = pred_df['label'].apply(lambda x: 'Instructions' if x == 0 else 'Recipe')
    pred_df['pred_label'] = pred_df['proba'].apply(lambda x: 'Recipe' if x > THRESHOLD else 'Instructions')
    logging.info(f'{model}')
    logging.info(pred_df[['text', 'proba']])
    logging.info(pred_df[['label', 'pred_label']])

    score = new_model1.evaluate([tfidf_one_page, X_meta_one_page], y_one_page,
                                batch_size=BATCH_SIZE, verbose=1)

    logging.info(f'Model Loss score: {round(score[0], 4)}')
    logging.info(f'Model Recall score: {round(score[1], 4)}')
    logging.info(f'Model Precision score: {round(score[2], 4)}')
    logging.info(f'Model Accuracy Evaluation : {round(score[3], 4)}')
    logging.info(f'Model AUC Evaluation : {round(score[4], 4)}')


@profile
def get_model(tf_idf_train, X_meta_train, results,
              embedding_dimensions=EMBEDDING_DIM):
    """
    The function creates the model for 2 different input data:
    NLP set and additional features not NLP set
    Layers: Embedding - TFIDF MATRIX Use masking to handle the variable sequence lengths,
            BiLSTM,
            concatenation of 2 data types,
            expand dimension to use BiLSTM second time,
            BiLSTM,
            Dense/fully connected layer with activation function "relu",
            Dropout layer to avoid overfitting,
            Dense/fully connected layer with activation function "sigmoid"
            All hyper-parameters as constants are in config.py
    :params tf_idf_train: ndArray(ndArray(int)) - a set with tfidf vectors
    :params X_meta_train: ndArray(int))- a set with non-nlp features
    :params results: set{str} - word vocabulary of the train set (config.VOCAB_SIZE=2284)
    :params embedding_dimensions:int hyper-parameter, can be done as = int(len(results)**0.25)
    :return a model
    """
    tf_idf_input = Input(shape=(tf_idf_train.shape[1],))
    meta_input = Input(shape=(X_meta_train.shape[1],))
    emb = Embedding(output_dim=embedding_dimensions,
                    input_dim=len(results) + 1,
                    input_length=tf_idf_train.shape[1],
                    mask_zero=True)(tf_idf_input)  # Use masking to handle the variable sequence lengths
    nlp_out = Bidirectional(LSTM(64))(emb)  #
    concat = tf.concat([nlp_out, meta_input], axis=1)
    concat = tf.expand_dims(concat, axis=-1)  # expand dimension to use bilstm second time
    concat_lstm = Bidirectional(LSTM(64))(concat)
    classifier = Dense(32, activation='relu')(concat_lstm)
    drop = Dropout(0.2)(classifier)  # to avoid overfitt
    output = Dense(1, activation='sigmoid')(drop)
    model = Model(inputs=[tf_idf_input, meta_input], outputs=[output])

    return model


@profile
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
                          metrics=[tf.keras.metrics.Recall(),
                                   tf.keras.metrics.Precision(),
                                   'accuracy',
                                   tf.keras.metrics.AUC()])
    es = tf.keras.callbacks.EarlyStopping()
    history = concat_biLstm.fit([tf_idf_train, X_meta_train],
                                y_train,
                                batch_size=BATCH_SIZE,
                                epochs=EPOCHS,
                                validation_split=0.2,
                                callbacks=[es])
    # evaluate the model
    score = concat_biLstm.evaluate([tf_idf_test, X_meta_test], y_test, batch_size=BATCH_SIZE)
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
    plt.title('The Loss vs val_Loss biLSTM Concat, activation = relu')
    plt.legend(loc='best')
    plt.show()

    concat_biLstm.save(MODEL_NAME)
    logging.info(f'The model is saved to {MODEL_NAME}')


if __name__ == '__main__':
    model_train()
