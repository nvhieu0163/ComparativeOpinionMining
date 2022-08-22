from modules.feature_selection import feature_select_chi2
from config import *
from modules.load_data.load_task_data import load_IBO_task_data
from modules.preprocessing.tag_IBO import tag_IBO_POS_for_each_word, tag_IBO_for_stc

if __name__ == '__main__':
    
    # Get top words feature using chi2 for CSI task
    feature_select_chi2.run_feature_selection_chi2(INPUT_LIST, CSI_LABEL_LIST, CSI_TRAIN_PATH, 'CSI')


    # Get top words feature using chi2 for AC task
    feature_select_chi2.run_feature_selection_chi2(INPUT_LIST, AC_LABEL_LIST, AC_TRAIN_PATH, 'AC')


    #Get top words feature using Chi2 for SP task
    feature_select_chi2.run_feature_selection_chi2(INPUT_LIST, SP_LABEL_LIST, SP_TRAIN_PATH, 'SP')


    #IBO and POS tagging for NER task
    IBO_data = load_IBO_task_data('./dataset/input/test_IBO.csv',
                    stc_idx_col_name='sentence_idx',
                    stc_col_name='stc',
                    sub_col_name='subject',
                    obj_col_name='object')

    tag_IBO_for_stc(IBO_data, True)
    
    IBO_data2 = load_IBO_task_data('./dataset/input/test_IBO.csv',
                    stc_idx_col_name='sentence_idx',
                    stc_col_name='stc',
                    sub_col_name='subject',
                    obj_col_name='object')

    tag_IBO_POS_for_each_word(IBO_data2, True)