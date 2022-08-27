
#config of project

CSI_TRAIN_PATH = './dataset/input/CSI/train_cs.csv'
CSI_TEST_PATH = './dataset/input/CSI/test_cs.csv'

AC_TRAIN_PATH = './dataset/input/AC/train_ac.csv'
AC_TEST_PATH = './dataset/input/AC/test_ac.csv'

SP_TEST_PATH = './dataset/input/SP/test_sc.csv'
SP_TRAIN_PATH = './dataset/input/SP/train_sc.csv'
SP_ENCODE = { 1: 'Negative', 2: 'Normal', 3: 'Positive' }

# Not found now. csv file has columns = "subject", 'object"
INPUT_IBO_TAG_PATH = './dataset/data_IBO.csv' 
IBO_ENCODE_DICT = {'O': 0, 'B-SUB': 1, 'I-SUB': 2, 'B-OBJ': 3, 'I-OBJ': 4}

#have POS and Tag for each word
NER_TRAIN_PATH = './dataset/input/NER/train_ner.csv'
NER_TEST_PATH = './dataset/input/NER/test_ner.csv'
NER_LABELS = ['B-SUB', 'I-SUB', 'B-OBJ', 'I-OBJ']

# This is the sentence input column name for CSI, AC and SP task, 
# because the column name of those is the same.
INPUT_LIST = ['stc']

#label list column of task
CSI_LABEL_LIST = ['label']
AC_LABEL_LIST = ['sức_mạnh', 'thiết_kế', 'giá', 'tính_năng', 'an_toàn']
SP_LABEL_LIST = ['sức_mạnh', 'thiết_kế', 'giá', 'tính_năng', 'an_toàn']

