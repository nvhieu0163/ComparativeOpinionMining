import datetime
import pandas as pd
import numpy as np

from imblearn.over_sampling import RandomOverSampler
from lightgbm import LGBMClassifier

from model.CSImodel import ComparativeSentenceModel
from modules.load_data.CSI_AC_SP_load_data import load_stc_data

from config import CSI_TEST_PATH, CSI_TRAIN_PATH


def train_CSI_chi2(k_best, X_train, y_train, model, oversampling=True, is_save_model = False):
    #set vocab from vocab_path with parameter: k_best
    model.set_vocab_chi2(k_best)
    
    #get label
    _y_train = [i.score for i in y_train]
    
    print('* Representing feature embedding using Chi2 .... ')
    _X_train = model.represent_feature_chi2(X_train)
    print('  Representing feature embedding using Chi2 DONE!')
    
    if oversampling:
        print('* RandomOverSampling ...')
        ros = RandomOverSampler(random_state=42, sampling_strategy=0.4)
        _X_train, _y_train = ros.fit_resample(_X_train, _y_train)
        print('  RandomOverSampling DONE!')

    print('* Trainning ...')
    model.train(_X_train, _y_train)
    print('  Trainning DONE!')

    if is_save_model:
        model.save('label')


def evaluate_CSI_chi2(X_test, y_test, model, tunning_threshold=True, save_result=True):
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
    
    print('=' * 20 + ' Performance of CSI model ' + (50 - len(' Performance of CSI model ')) * '=')
    print("- K_best       :", model.k_best)
    print("- Threshold    :", model.threshold)
    print("- F1 score     :", f1)
    print("- Precision    :", p)
    print("- Recall       :", r)

    if save_result:
        result = pd.DataFrame({'score': [model.model, model.k_best, model.threshold, f1, p, r]},
                          index=['Model', 'K_best', 'Threshold', 'F1', 'Precision', 'Recall'])

        result.to_csv('./dataset/output/evaluate/CSI_chi2_result_{}_oversample.csv'.format(
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
    

if __name__ == '__main__':

    model = LGBMClassifier()

    X_train, y_train = load_stc_data(path=CSI_TRAIN_PATH,
                                    stc_index_col_name='sentence_idx',
                                    stc_col_name='stc',
                                    label_col_name='label')

    X_test, y_test = load_stc_data(path=CSI_TEST_PATH,
                                    stc_index_col_name='sentence_idx',
                                    stc_col_name='stc',
                                    label_col_name='label')
                                    
    vocab_path = './dataset/output/chi2_score_dict/CSI_stc_label.csv'
    CSI_model = ComparativeSentenceModel(task_name = 'CSI', vocab_path = vocab_path, base_model = model)
    
    # tuning k_best
    # k_list = [500, 1000, 1500, 2000, 2500, 3000]
    
    # tune k_best
    # k_best = tune_k_best(k_list, X_train, y_train, X_test, y_test, CSI_model)
    
    # train model
    train_CSI_chi2(2000, X_train, y_train, CSI_model, oversampling=True, is_save_model = True)
    evaluate_CSI_chi2(X_test, y_test, CSI_model, tunning_threshold = True, save_result = True)