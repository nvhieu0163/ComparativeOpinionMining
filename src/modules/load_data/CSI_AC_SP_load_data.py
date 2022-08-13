from typing import List
import pandas as pd

from classes.CSI_AC_SP_classes import InputObject, OutputObject

# CSI TASK
# load file need 1 column input, 1 column output
def load_stc_data(
        path: str,
        stc_index_col_name: str,
        stc_col_name: str,
        label_col_name: str) -> object:
    
    inputs = []
    outputs = []

    df = pd.read_csv(path)
    df = df.reset_index()

    for _, row in df.iterrows():
        stc_idx = row[stc_index_col_name]
        stc = row[stc_col_name].strip()
        label = row[label_col_name]

        inputs.append(InputObject(stc_idx, stc))
        outputs.append(OutputObject(label))

    return inputs, outputs


# AC TASK
# load file need 1 column input, multi column output
# return 1 list of sentence
# return list of len(multilabel list) lists.
def load_stc_multilabel_data(
        path: str,
        stc_index_col_name: str,
        stc_col_name: str,
        label_col_names: list) -> object:

    inputs = []
    outputs = [ [] for i in range(len(label_col_names))]

    df = pd.read_csv(path)
    df = df.reset_index()

    for _, row in df.iterrows():
        stc_idx = row[stc_index_col_name]
        stc = row[stc_col_name].strip()
        
        inputs.append(InputObject(stc_idx, stc))

        count = 0
        for label in label_col_names:
            
            _label = row[label] #lấy giá trị tương ứng với label

            out_obj = OutputObject(_label) #tạo object đầu ra

            outputs[count].append(out_obj)

            count += 1

    return inputs, outputs


def load_polarity_data(
        path: str,
        stc_idx_col_name: str,
        stc_col_name: str,
        label_col_names: List) -> object:
    
    inputs = [ [] for i in range(len(label_col_names))]
    outputs = [ [] for i in range(len(label_col_names))]

    df = pd.read_csv(path)
    df = df.reset_index()

    for _,row in df.iterrows():
        idx_label = 0
        for label in label_col_names:
            if int(row[label]) != 0 :
                stc_idx = row[stc_idx_col_name]
                stc = row[stc_col_name].strip()

                inputs[idx_label].append(InputObject(stc_idx, stc))
                outputs[idx_label].append(OutputObject(row[label]))

            idx_label += 1


    return inputs, outputs