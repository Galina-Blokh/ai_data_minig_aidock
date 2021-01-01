import tensorflow

from config import BATCH_SIZE, THRESHOLD


def eval_on_one_page(sent2vec_one_page, X_meta_one_page, y_one_page, model,text,remove_stop_words_text):
    """Load model from file and evaluate on data from one page
    : params sent2vec_one_page: ndArray,
            X_meta_one_page: ndArray,
            y_one_page: ndArray,
            model:str - path to the model
    """

    new_model1 = tensorflow.keras.models.load_model(model)
    preds = new_model1.predict([sent2vec_one_page, X_meta_one_page])
    # THRESHOLD = 0.5
    import pandas as pd
    pred_df = pd.DataFrame(columns=['text', 'pred_label'])
    pred_df['text'] = text
    pred_df['remove_stop_words_text'] = remove_stop_words_text

    pred_df['pred_label'] = preds
    pred_df['label'] = y_one_page
    pred_df['label'] = pred_df['label'].apply(lambda x: 'Instructions' if x == 0 else 'Recepie')
    pred_df['pred_label'] = pred_df['pred_label'].apply(lambda x: 'Recepie' if x > THRESHOLD else 'Instructions')
    print(pred_df)

    score1 = new_model1.evaluate([sent2vec_one_page, X_meta_one_page], y_one_page,
                                 batch_size=BATCH_SIZE, verbose=1)
    print(f'{model} Loss score : {round(score1[0], 2)}')
    print(f'{model} Accuracy Evaluation : {round(score1[1], 2)}')
