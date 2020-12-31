import tensorflow
import matplotlib.pyplot as plt
import pandas as pd
from config import BATCH_SIZE, EPOCHS, MODEL_NAME
from preprocess import get_model, sent2vec

if __name__ == '__main__':
    train_data_clean = pd.read_pickle('C:\\Users\\galin\\PycharmProjects\\aiAssignment\\data\\train_data_clean.pkl')
    print('in train', round(train_data_clean['label'].value_counts() / len(train_data_clean) * 100, 2))

    test_data_clean = pd.read_pickle('C:\\Users\\galin\\PycharmProjects\\aiAssignment\\data\\test_data_clean.pkl')
    print('in test', round(test_data_clean['label'].value_counts() / len(test_data_clean) * 100, 2))

    # max len sequence count (it is 121 in train - we will use it)
    max_sequence_length = train_data_clean['clean_paragraph_len'].max()
    print(max_sequence_length)
    # vocab_size count in train set
    results = set()
    train_data_clean.remove_stop_words.str.split().apply(results.update)
    vocab_size = len(results)
    print(vocab_size)

    # sent to sequence only for  NLP TRAIN
    sent2vec_train = sent2vec(train_data_clean.remove_stop_words, max_sequence_length, vocab_size)
    # for other features train
    X_meta_train = train_data_clean[['sent_count', 'num_count', 'clean_paragraph_len', 'contains_pron']]
    y_train = train_data_clean['label']

    # for NLP TEST
    sent2vec_test = sent2vec(test_data_clean.remove_stop_words, max_sequence_length, vocab_size)
    # for other features test
    X_meta_test = test_data_clean[['sent_count', 'num_count', 'clean_paragraph_len', 'contains_pron']]
    y_test = test_data_clean['label']

    # create and train the MODEL
    concat_biLstm = get_model(sent2vec_train, X_meta_train, results)
    concat_biLstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    es = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min')
    history = concat_biLstm.fit([sent2vec_train, X_meta_train],
                                y_train,
                                batch_size=BATCH_SIZE,
                                epochs=EPOCHS,
                                validation_split=0.2,
                                callbacks=[es])
    # evaluate the model
    score = concat_biLstm.evaluate([sent2vec_test, X_meta_test], y_test, batch_size=BATCH_SIZE, verbose=1)
    print(u'Model Loss score: {}'.format(score[0]))
    print(u'Model Accuracy Evaluation : {}'.format(score[1]))

    # plot the loss
    plt.figure(figsize=(15, 4))
    plt.plot(history.history['loss'], 'r', label=f'loss  {round(score[0], 2)}')
    plt.plot(history.history['val_loss'], 'g--', label=f'val_loss  {round(score[1], 2)}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('The Loss vs val_Loss LSTM Concat, activation = relu')
    plt.legend(loc='best')
    plt.show()

    concat_biLstm.save(MODEL_NAME)

