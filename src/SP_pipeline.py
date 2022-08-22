from collections import Counter
import datetime
import pandas as pd
import numpy as np

from imblearn.over_sampling import RandomOverSampler
from lightgbm import LGBMClassifier

from model.SP_model import SentimentPolarizationModel 
from modules.load_data.load_task_data import load_polarity_data

from config import SP_TEST_PATH, SP_TRAIN_PATH, SP_LABEL_LIST


def train_SP_chi2(aspect, k_best, X_train, y_train, model, oversampling=True, is_save_model = False):
    #set vocab from vocab_path with parameter: k_best
    model.set_vocab_chi2(k_best)
    
    #get label
    _y_train = [i.score for i in y_train]
    
    print('* Representing feature embedding using Chi2 .... ')
    _X_train = model.represent_feature_chi2(X_train)
    print('  Representing feature embedding using Chi2 DONE!')
    
    if oversampling:
        print('* RandomOverSampling ...')
        ros = RandomOverSampler(random_state=42)
        _X_train, _y_train = ros.fit_resample(_X_train, _y_train)
        print(Counter(_y_train))
        print('  RandomOverSampling DONE!')

    print('* Trainning ...')
    model.train(_X_train, _y_train)
    print('  Trainning DONE!')

    if is_save_model:
        model.save(aspect)


def evaluate_SP_chi2(aspect, X_test, y_test, model, tunning_threshold=True, save_result=True):
    #get label
    _y_test = [i.score for i in y_test]
    
    # Representing feature embedding using Chi2
    _X_test = model.represent_feature_chi2(X_test)
    
    if tunning_threshold:
        model.set_optimal_threshold(_X_test, _y_test)
    
    #predcit
    predict = model.predict(_X_test)
    # test_df['predict0'] = predict
    # test_df.to_csv('data/output/test_data/CSI/chi2_test_k_{}.csv'.format(k_best))
    
    #evaluate
    neg_p, neg_r, neg_f1 = model.get_evaluate(_y_test, predict, 1)
    neu_p, neu_r, neu_f1 = model.get_evaluate(_y_test, predict, 2)
    pos_p, pos_r, pos_f1 = model.get_evaluate(_y_test, predict, 3)
    
    print("- K_best       :", model.k_best)
    print("- F1 negative  :", neg_f1)
    print("- F1 positive  :", pos_f1)
    print("- F1 neutral   :", neu_f1)

    if save_result:
        result = pd.DataFrame({'score': [model.model, model.k_best, neg_f1, pos_f1, neu_f1]},
                          index=['Model', 'K_best', 'F1 negative', 'F1 positive', 'F1 neutral'])

        result.to_csv('./dataset/output/evaluate/SP_chi2_result_{m}_{a}_{b}.csv'.format(
            m = model.model,
            a = aspect,
            b = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))

    return neg_f1, pos_f1, neu_f1


if __name__ == '__main__':
    
    # load data
    # return: X_train: list of <list>:sentence for each aspect, Y_train: list of <list>:label for each aspect
    X_train, y_train = load_polarity_data(SP_TRAIN_PATH, 'sentence_idx', 'stc', SP_LABEL_LIST)
    X_test, y_test = load_polarity_data(SP_TEST_PATH, 'sentence_idx', 'stc', SP_LABEL_LIST)

    #set parameters
    k_best = 3000
    base_model = LGBMClassifier()

    #
    neg_f1 = []
    pos_f1 = []
    neu_f1 = []

    # Training model for each label
    idx_label = 0
    for aspect in SP_LABEL_LIST:
        print('=' * 20 + ' Training SP with chi2 '+ aspect + ' ' + (60 - len(' Training SP with chi2 ' + aspect)) * '=')
        
        vocab_path = './dataset/output/chi2_score_dict/SP_stc_{}.csv'.format(aspect)
        model = SentimentPolarizationModel('SP', vocab_path, base_model)
        
        train_SP_chi2(aspect, 3000, X_train[idx_label], y_train[idx_label], model, True, True)
        f1, f2, f3 = evaluate_SP_chi2(aspect, X_test[idx_label], y_test[idx_label], model, False, True)

        neg_f1.append(f1)
        pos_f1.append(f2)
        neu_f1.append(f3)

        idx_label += 1
    
    macro_neg_f1 = np.array(neg_f1).mean()
    macro_pos_f1 = np.array(pos_f1).mean()
    macro_neu_f1 = np.array(neu_f1).mean()

    print('=' * 20 + ' Performance of SP model ' + (60 - len(' Performance of SP model ')) * '=')
    print("- Macro-F1 negative       :", macro_neg_f1)
    print("- Macro-F1 positive       :", macro_pos_f1)
    print("- Macro-F1 neutral        :", macro_neu_f1)

    result = pd.DataFrame({'score': [model.model, k_best, macro_neg_f1, macro_pos_f1, macro_neu_f1]},
                          index=['Model', 'K_best', 'Macro-F1 negative', 'Macro-F1 positive', 'Macro-F1 neutral'])
    result.to_csv(
        './dataset/output/evaluate/SP_chi2_result_{m}_overall_{d}.csv'.format(
            m=model.model,
            d=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))