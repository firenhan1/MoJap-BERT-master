import argparse
import os
import pickle 
from glob import glob
from tqdm import tqdm
from transformers import (
    BertForMaskedLM
)
from transformers import BertJapaneseTokenizer
import torch
import math
import numpy as np
import operator

def main(args):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input_file', required=True, metavar='PATH')
    # parser.add_argument('--model_path', required=True, metavar='PATH')

    

    # args = parser.parse_args()

    print(glob(args.input_file))
    for file in glob(args.input_file):
        newList = []
        fileName = os.path.basename(file)
        baseName = os.path.splitext(fileName)[0]
        extension = os.path.splitext(fileName)[1]
        new_file = args.output_dir + baseName + '_best_candidate' + extension
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
                candidateData = (title, score_ocr, sentence_with_space)
                if (len(candidate_list) == 0 or candidate_list[0][0] == candidateData[0]):
                    candidate_list.append(candidateData)
                else:
                    best_candidate = max(candidate_list, key=operator.itemgetter(1))
                    best_candidate_string = ' '.join([best_candidate[0], best_candidate[2]]) # [title, sentence]
                    best_candidate_string = best_candidate_string + '\n'
                    newList.append(best_candidate_string)
                    del best_candidate_string
                    del best_candidate
                    del candidate_list[:]
                    candidate_list.append(candidateData)
                # newStr = ' '.join(newArr)
                # newStr = newStr + '\n'
                # f1.write(newStr)
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
                


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, metavar='PATH')
    parser.add_argument('--output_dir', required=True, metavar='PATH')
    args = parser.parse_args()
    main(args)

    

    
