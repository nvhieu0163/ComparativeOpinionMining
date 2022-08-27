import datetime
import pandas as pd
import sklearn_crfsuite

from sklearn_crfsuite.metrics import flat_classification_report

#module for preprocessing for NER task
from modules.preprocessing.featuring_NER import *
from classes.preprocessing_class import NER_SentenceGetter

#config to load data
from config import NER_TRAIN_PATH, NER_TEST_PATH

#config to get labels
from config import NER_LABELS

#NER_object: [(Word, POS, Tag), ...]
def get_NER_Objects(df_path: str):
    df = pd.read_csv(df_path)

    getter =  NER_SentenceGetter(df)
    ner_objects_list = getter.sentences
    return ner_objects_list


def get_feature(ner_objects_list):
    return [sent2features(s) for s in ner_objects_list]


def get_label(ner_objects_list):
    return [sent2labels(s) for s in ner_objects_list]


def get_stc(ner_objects_list):
    return [sent2tokens(s) for s in ner_objects_list]


def train_NER_crf(train_object_list):
    X_train = get_feature(train_object_list)
    y_train = get_label(train_object_list)

    crf = sklearn_crfsuite.CRF(algorithm='lbfgs',
          c1 = 0.1,
          c2 = 0.1,
          max_iterations = 100,
          all_possible_transitions = True)

    crf.fit(X_train, y_train)

    return crf


def evaluate_NER(model, test_object_list, isExportDifferency: bool):
    X_test = get_feature(test_object_list)
    y_test = get_label(test_object_list)
    
    y_pred = model.predict(X_test)

    #print report
    print(flat_classification_report(y_test, y_pred, labels = NER_LABELS))
    
    #save report
    output = flat_classification_report(y_true = y_test, y_pred = y_pred, labels = NER_LABELS, output_dict = True)
    df = pd.DataFrame(output).transpose()
    df = df.round(4)
    df.to_csv('./dataset/output/evaluate/NER_result_{a}_{b}.csv'.format(
        a = 'CRF',
        b = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))

    if isExportDifferency:
        pass


if __name__ == '__main__':
    train_objects = get_NER_Objects(NER_TRAIN_PATH)
    test_objects = get_NER_Objects(NER_TEST_PATH)

    print('=' * 20 + ' Training NER with CRF ' + (60 - len(' Training NER with CRF ')) * '=')
    trained_model = train_NER_crf(train_object_list = train_objects)
    print('Training Done!!!')

    print('=' * 20 + ' Performance of NER model ' + (60 - len(' Performance of NER model ')) * '=')
    evaluate_NER(trained_model, test_objects, isExportDifferency= False)





