from datetime import datetime
from typing import List
import pandas as pd

from pyvi import ViTokenizer, ViPosTagger

from modules.load_data import write_data
from classes.preprocessing_class import IBOInputObject
from config import IBO_ENCODE_DICT


def stc_to_word_list(input_stc: str) -> list:
    return input_stc.split(' ')


def tokenize_stc(stc: str):
    return ViTokenizer.tokenize(stc)


def pos_tag_stc(tokenzied_stc: str):
    pos_tags = ViPosTagger.postagging(tokenzied_stc)
    return pos_tags[1]


def tokenize_InputObject(input: IBOInputObject):
    
    tokenized_object = input

    tokenized_object.stc = tokenize_stc(input.stc)
    tokenized_object.object = tokenize_stc(input.object)
    tokenized_object.subject = tokenize_stc(input.subject)

    return tokenized_object


#get IBO tag list for 1 input Object: IBOInputObject
def tag_IBO_InputObject(tokenized_input: IBOInputObject):

    output = tokenized_input
    # CREATE IB TAGGED SENTENCE
    stc_contain_IB = output.stc

    # TAG IB SUBJECT
    if output.subject != 'undentified':
        stc_contain_IB = output.stc.replace(output.subject, 'B-SUB' + ' I-SUB' * (len(output.subject.split(' ')) - 1 ))

        # if tokenized subject and tokenized subject in sentence are differents
        if 'B-S' not in stc_contain_IB:

            splited_subject = output.subject.split('_')
            subject_words = []

            for chunk in splited_subject:
                subject_words = subject_words + chunk.split(' ')
            
            stc_contain_IB = stc_contain_IB.replace(subject_words[0], 'B-SUB')

            for word in subject_words[1:]:
                stc_contain_IB.replace(word, 'I-SUB')

    # TAG IB OBJECT
    if output.object != 'undentified':
        stc_contain_IB = stc_contain_IB.replace(output.object, 'B-OBJ' + ' I-OBJ' * (len(output.object.split(' ')) - 1))

        # if tokenized object and tokenized object in sentence are differents
        if 'B-O' not in stc_contain_IB:

            splited_object = output.object.split('_')
            object_words = []

            for chunk in splited_object:
                object_words = object_words + chunk.split(' ')
            
            stc_contain_IB = stc_contain_IB.replace(object_words[0], 'B-OBJ')

            for word in object_words[1:]:
                stc_contain_IB.replace(word, 'I-OBJ')

    # TAG 'O' FOR OTHER WORD
    stc_contain_IB_list = stc_to_word_list(stc_contain_IB) 

    #output IBO list for each sentence
    IBO_stc = ['O' if word not in ['B-SUB', 'I-SUB', 'B-OBJ', 'I-OBJ'] else word for word in stc_contain_IB_list]
    
    return [IBO_ENCODE_DICT[i] for i in IBO_stc]


#tag IBO for list of input object
def tag_IBO_for_stc(inputs: List[IBOInputObject], save = True):
    
    tokenized_stcs = []
    IBO_tags = []
    stc_idxs = []

    for input in inputs:
        #tokenize
        _input = tokenize_InputObject(input)

        #get tag
        tag = tag_IBO_InputObject(_input)
        
        # convert stc to word list
        _input.stc = stc_to_word_list(_input.stc)

        stc_idxs.append(_input.stc_idx)
        tokenized_stcs.append(_input.stc)
        IBO_tags.append(tag)

    if save:
        write_data.export_to_file('./dataset/output/IBO_tag_data/IBO_tag_stc_data_{}.txt'.format(
            datetime.now().strftime('%Y%m%d_%H%M%S')), tokenized_stcs, IBO_tags, stc_idxs)


#tag IBO for list of input object
def tag_IBO_POS_for_each_word(inputs: List[IBOInputObject], save = True):
    
    word_idx_list = []
    word_list = []
    pos_tags = []  # 1 element is for each word
    IBO_tags = []  # 1 element is for each word

    for input in inputs:
        #tokenize
        _input = tokenize_InputObject(input)

        #get tag
        tag = tag_IBO_InputObject(_input)

        #get pos
        pos = pos_tag_stc(_input.stc)

        # convert stc to word list
        _input.stc = stc_to_word_list(_input.stc)

        for i in range(0, len(_input.stc)):
            word_idx_list.append(_input.stc_idx)
            word_list.append(_input.stc[i])
            pos_tags.append(pos[i])
            IBO_tags.append(tag[i])

    if save:
        IBO_tag_data_df = pd.DataFrame({'Word_idx': word_idx_list,
                                        'Word': word_list,
                                        'POS': pos_tags,
                                        'Tag': IBO_tags,
                                        })
                        
        IBO_tag_data_df.to_csv(
            './dataset/output/IBO_tag_data/IBO_tag_word_data_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)