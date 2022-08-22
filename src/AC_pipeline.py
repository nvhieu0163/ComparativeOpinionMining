import datetime
import pandas as pd
import numpy as np

from imblearn.over_sampling import RandomOverSampler
from lightgbm import LGBMClassifier

from model.ACmodel import AspectClassificationModel
from modules.load_data.load_task_data import load_stc_multilabel_data

from config import AC_TEST_PATH, AC_TRAIN_PATH, AC_LABEL_LIST


def train_AC_chi2(aspect, k_best, X_train, y_train, model, oversampling=True, is_save_model = False):
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
        print('  RandomOverSampling DONE!')

    print('* Trainning ...')
    model.train(_X_train, _y_train)
    print('  Trainning DONE!')

    if is_save_model:
        model.save(aspect)


def evaluate_AC_chi2(aspect, X_test, y_test, model, tunning_threshold=True, save_result=True):
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
    p, r, f1 = model.get_evaluate(_y_test, predict)
    
    print("- K_best       :", model.k_best)
    print("- Threshold    :", model.threshold)
    print("- F1 score     :", f1)
    print("- Precision    :", p)
    print("- Recall       :", r)

    if save_result:
        result = pd.DataFrame({'score': [model.model, model.k_best, model.threshold, f1, p, r]},
                          index=['Model', 'K_best', 'Threshold', 'F1', 'Precision', 'Recall'])

        result.to_csv('./dataset/output/evaluate/AC_chi2_result_{m}_{a}_{b}_oversample.csv'.format(
            m = model.model,
            a = aspect,
            b = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))

    return p, r, f1


if __name__ == '__main__':
    
    # load data
    # return: X_train: List of sentence, Y_train: list of <list>: label for each aspect
    X_train, y_train = load_stc_multilabel_data(AC_TRAIN_PATH, 'sentence_idx', 'stc', AC_LABEL_LIST)
    X_test, y_test = load_stc_multilabel_data(AC_TEST_PATH, 'sentence_idx', 'stc', AC_LABEL_LIST)

    #set parameters
    k_best = 3000
    base_model = LGBMClassifier()

    #
    f1 = []
    p = []
    r = []

    # Training model for each label
    idx_label = 0
    for aspect in AC_LABEL_LIST:
        print('=' * 20 + ' Training AC with chi2 '+ aspect + ' ' + (60 - len(' Training AC with chi2 ' + aspect)) * '=')
        
        vocab_path = './dataset/output/chi2_score_dict/AC_stc_{}.csv'.format(aspect)
        model = AspectClassificationModel('AC', vocab_path, base_model)
        
        train_AC_chi2(aspect, 3000, X_train, y_train[idx_label], model, True, True)
        _p, _r, _f1 = evaluate_AC_chi2(aspect, X_test, y_test[idx_label], model, True, True)

        f1.append(_f1)
        p.append(_p)
        r.append(_r)

        idx_label += 1
    
    macro_f1 = np.array(f1).mean()
    macro_p = np.array(p).mean()
    macro_r = np.array(r).mean()

    print('=' * 20 + ' Performance of AC model ' + (60 - len(' Performance of AC model ')) * '=')
    print("- Macro-F1           :", macro_f1)
    print("- Macro-P            :", macro_p)
    print("- Macro-R            :", macro_r)

    result = pd.DataFrame({'score': [base_model, k_best, macro_f1, macro_p, macro_r]},
                          index=['Model', 'K_best', 'Macro-F1', 'Macro-P', 'Macro-R'])
    result.to_csv(
        './dataset/output/evaluate/AC_chi2_result_{m}_overall_{d}.csv'.format(
            m=base_model,
            d=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))