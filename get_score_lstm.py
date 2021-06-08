import argparse
import os
import pickle 
from glob import glob
from tqdm import tqdm
import torch
import math
import numpy as np
import operator
from keras.models import Sequential, load_model
# from keras.utils import multi_gpu_model
import h5py
import numpy as np
import pandas as pd
import re
from pickle import load
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
from tensorflow.compat.v1.keras.backend import set_session
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer





def main(args):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input_file', required=True, metavar='PATH')
    # parser.add_argument('--model_path', required=True, metavar='PATH')

    ###############################   UTILS FUNCTION  ############################
    def text_cleaner(text):
        # lower case text
        newString = text.lower()
        newString = re.sub(r"'s\b","",newString)
        newString = re.sub("[^a-zA-ZạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ]", " ", newString)
        long_words=[]
        # remove short word
        for i in newString.split():
            if len(i) >= 1:
                long_words.append(i)
        return (" ".join(long_words)).strip()

    def encode_string(mapping, seq_length, in_text):
        encoded = [mapping[word] for word in in_text.split()]
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        return encoded

    def decode_string(mapping, in_text):
        out_text = ""
        for i in range(len(in_text.split())):
            for char, index in mapping.items():
                if index == in_text[i]:
                    out_text += char
                    break
        return out_text

    def insert(source_str, insert_str, pos):
        return source_str[:pos]+insert_str+source_str[pos:]

    def replace(source_str, insert_str, start_pos):
        source_list = list(source_str)
        if (start_pos > len(source_list)):
            return source_str
        for i in range(len(insert_str)):
            source_list[start_pos + i] = insert_str[i]
        return ''.join(source_list)

    # Lấy xác suất của 1 từ kế tiếp 1 từ
    def get_probability_of_given_next_word(model, mapping, SEQ_LENGTH, seed_text, wordSecond):
        try:
            in_text = seed_text
            out_text = in_text[:]
            out_word_predict_encode = encode_string(mapping, SEQ_LENGTH, out_text)
            proba_list_word = model.predict_proba(out_word_predict_encode)
            return proba_list_word[0][mapping[wordSecond]] #float
        except:
            return 0

    # # Lấy xác suất của 1 từ kế tiếp 1 từ
    # def get_probability_of_given_next_word(model, mapping, SEQ_LENGTH, seed_text, wordSecond):
    #   in_text = text_cleaner(seed_text)
    #   out_text = in_text[:]
    #   out_word_predict_encode = encode_string(mapping, SEQ_LENGTH, out_text)
    #   proba_list_word = model.predict_proba(out_word_predict_encode)
    #   return proba_list_word[0][mapping[wordSecond]] #float


    # Hàm tính điểm 1 câu
    def get_score_one_sequence(model, mapping, sentence):
    # print('getScore_sentence=', sentence)
        proba_res = 0
        sentenceChars = sentence.split()
        lenSentences = len(sentenceChars) - 1
        while lenSentences > 0:
            wordLast = ''.join(sentenceChars[-1])
            sentenceFirst = ' '.join(sentenceChars[0:lenSentences])
            # print('sentenceFirst=', sentenceFirst, 'wordLast=', wordLast)
            score = get_probability_of_given_next_word(model, mapping, SEQ_LENGTH, sentenceFirst, wordLast)
            if (score == 0):
                score = 1e-100
            # print('sentenceFirst=', sentenceFirst, 'wordLast=', wordLast, 'score=', score, 'log=', np.log(score))
            proba_res = proba_res + np.log(score)
            sentenceChars.pop(lenSentences)
            lenSentences = lenSentences - 1
        # print('len(sentenceChars)=', len(sentenceChars), 'proba_res=', proba_res)
        lenSentence = len(sentence.split())
        if (lenSentence > 0):
            proba_res = proba_res / len(sentence.split())
        return proba_res



    ###################################
    #FUNCTION FOR PREDICT NEXT word
    ###################################
    def next_word_proba(model, mapping, seed_text):
        in_text = seed_text
        out_text = in_text[:]
        i = len(in_text)

        out_text_predict_encode = encode_string(mapping , SEQ_LENGTH , out_text)
        proba_list_word = model.predict_proba(out_text_predict_encode)
        index_word_list = np.argsort(proba_list_word[0])[-5:][::-1]

        proba_results = proba_list_word[0, index_word_list]

        word_result = [idx_word_mapping[idx] for idx in index_word_list]

        return word_result, proba_results

    def get_max_proba(model, mapping, seed_text):
        in_text = seed_text
        out_text = in_text[:]
        i = len(in_text)

        out_text_predict_encode = encode_string(mapping , SEQ_LENGTH , out_text)
        predict_res = model.predict(out_text_predict_encode)
        next_word_predict = np.argmax(predict_res, axis=1)

        max_proba = np.max(predict_res)
        return next_word_predict, max_proba



    def next_char_multi_choice(model, mapping, seed_text):
        in_text = text_cleaner(seed_text)
        out_text = in_text[:]
        i = len(in_text)

        out_text_predict_encode = encode_string(mapping , SEQ_LENGTH , out_text)
        proba_list_word = model.predict_proba(out_text_predict_encode)
        index_word_list = np.argsort(proba_list_word[0])[-5:][::-1]

        proba_results = proba_list_word[0, index_word_list]

        word_result = [idx_word_mapping[idx] for idx in index_word_list]

        return word_result, proba_results


    # args = parser.parse_args()
    model = load_model(args.model_path)
    mapping = load(open(args.mapping_file, 'rb'))
    mapping = mapping.word_index
    vocab_size = len(mapping)
    print('Vocabulary Size: %d' % vocab_size)
    print("Number of layers: %d" % len(model.layers))
    print(glob(args.input_file))
    SEQ_LENGTH = int(args.seq_length)
    CORRECT_THRESHOLD = 0.001
    idx_word_mapping = dict([(value, key) for key, value in mapping.items()]) 


    for file in glob(args.input_file):
        newList = []
        fileName = os.path.basename(file)
        baseName = os.path.splitext(fileName)[0]
        extension = os.path.splitext(fileName)[1]
        # lamda_value_list = np.arange(0.5, 1.0, 0.1) 
        lamda_value_list = [0.5]
        for lamda_value in lamda_value_list:
            new_file = args.output_dir + baseName + '_rescore' + '_' + str(int(lamda_value*10)) + extension
            # if not os.path.exists(new_file):
            #     with open(new_file, 'w', encoding='utf-8'): 
            #         pass
            
            print(new_file)
            with open(file, encoding='utf-8') as fo:
                k = [s for s in fo]
                newList = []
                candidate_list = []
                for line in k:
                    line = line.rstrip()
                    elements = line.split()
                    title = elements[0]
                    score_ocr = float(elements[1])
                    if len(elements) <= 2:
                        sentence_elements = [' ']
                        sentence = ''
                        sentence_with_space = ' '
                    else:
                        sentence_elements = elements[2:len(elements)]
                        sentence = ''.join(sentence_elements)
                        sentence_with_space = ' '.join(sentence_elements)
                    
                    if len(sentence_elements) >= 5:
                        score_lm = -get_score_one_sequence(model, mapping, sentence_with_space)
                        new_score = ((1 - lamda_value)*score_lm) + (lamda_value*score_ocr)
                        del score_lm
                    else:
                        new_score = score_ocr
                        
                    
                    candidateRescoreData = (title, new_score, sentence_with_space)
                    if (len(candidate_list) == 0 or candidate_list[0][0] == candidateRescoreData[0]):
                        candidate_list.append(candidateRescoreData)
                    else:
                        best_candidate = max(candidate_list, key=operator.itemgetter(1))
                        best_candidate_string = ' '.join([best_candidate[0], best_candidate[2]]) # [title, sentence]
                        best_candidate_string = best_candidate_string + '\n'
                        newList.append(best_candidate_string)
                        del best_candidate_string
                        del best_candidate
                        del candidate_list[:]
                        candidate_list.append(candidateRescoreData)
                    del new_score
                # handle to choose best candidate for the 10 last elements (end to loop, we still get 10 last candidates but there is no the next any element => candidate_list.length > 0) 
                if len(candidate_list) > 0:
                    best_candidate = max(candidate_list, key=operator.itemgetter(1))
                    best_candidate_string = ' '.join([best_candidate[0], best_candidate[2]]) # [title, sentence]
                    best_candidate_string = best_candidate_string + '\n'
                    newList.append(best_candidate_string)
                f1=open(new_file, 'w', encoding='utf-8')
                
                f1.writelines(newList)
                f1.close()
                fo.close()
                
        
    del tokenizer
    del bertMaskedLM


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, metavar='PATH')
    parser.add_argument('--model_path', required=True, metavar='PATH')
    parser.add_argument('--mapping_file', required=True, metavar='PATH')
    parser.add_argument('--output_dir', required=True, metavar='PATH')
    parser.add_argument('--seq_length', required=True)
    args = parser.parse_args()
    main(args)

    

    
