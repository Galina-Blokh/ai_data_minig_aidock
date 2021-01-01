import tensorflow

from config import BATCH_SIZE


def eval_on_one_page(sent2vec_one_page, X_meta_one_page, y_one_page, model):
    """Load model from file and evaluate on data from one page
    : params sent2vec_one_page: ndArray,
            X_meta_one_page: ndArray,
            y_one_page: ndArray
    """

    new_model1 = tensorflow.keras.models.load_model(model)
    score1 = new_model1.evaluate([sent2vec_one_page, X_meta_one_page], y_one_page,
                                 batch_size=BATCH_SIZE, verbose=1)
    print(f'{model} Loss score : {round(score1[0], 2)}')
    print(f'{model} Accuracy Evaluation : {round(score1[1], 2)}')
