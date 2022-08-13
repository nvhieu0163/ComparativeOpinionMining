
#config of project

CSI_TRAIN_PATH = './dataset/input/CSI/train_cs.csv'
CSI_TEST_PATH = './dataset/input/CSI/test_cs.csv'

AC_TRAIN_PATH = './dataset/input/AC/train_ac.csv'
AC_TEST_PATH = './dataset/input/AC/test_ac.csv'

SP_TEST_PATH = './dataset/input/SP/test_sc.csv'
SP_TRAIN_PATH = './dataset/input/SP/train_sc.csv'


# This is the sentence input for CSI, AC and SP task, because the column name of those is the same.
INPUT_LIST = ['stc']

CSI_LABEL_LIST = ['label']
AC_LABEL_LIST = ['sức_mạnh', 'thiết_kế', 'giá', 'tính_năng', 'an_toàn']
SP_LABEL_LIST = ['sức_mạnh', 'thiết_kế', 'giá', 'tính_năng', 'an_toàn']