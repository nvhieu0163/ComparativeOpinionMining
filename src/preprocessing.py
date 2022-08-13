from modules.feature_selection import feature_select_chi2
from config import *


if __name__ == '__main__':
    
    # Get top words feature using chi2 for CSI task
    feature_select_chi2.run_feature_selection_chi2(INPUT_LIST, CSI_LABEL_LIST, CSI_TRAIN_PATH, 'CSI')


    # Get top words feature using chi2 for AC task
    feature_select_chi2.run_feature_selection_chi2(INPUT_LIST, AC_LABEL_LIST, AC_TRAIN_PATH, 'AC')


    #Get top words feature using Chi2 for SP task
    feature_select_chi2.run_feature_selection_chi2(INPUT_LIST, SP_LABEL_LIST, SP_TEST_PATH, 'SP')


