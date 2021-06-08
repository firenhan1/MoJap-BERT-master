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

    f2=open(args.output_file, 'w', encoding='utf-8')
    with open(args.input_file, encoding='utf-8') as fo:
        result = [s for s in fo]
        newList = []
        candidate_list = []
        with open(args.candidate_file, encoding='utf-8') as f1:
            candidates = [i for i in f1]
            for line in result:
                result_line = line.rstrip()
                result_elements = result_line.split()
                result_title = result_elements[0]
                if len(result_elements) > 1:
                    result_sentence = result_elements[1:len(result_elements)]
                    for candidate_line in candidates:
                        line2 = candidate_line.rstrip()
                        candidate_elements = line2.split()
                        candidate_title = candidate_elements[0]
                        if result_title == candidate_title and len(candidate_elements) > 2:
                            score = candidate_elements[1]
                            candidate_sentence = candidate_elements[2:len(candidate_elements)]
                            if result_sentence == candidate_sentence:
                                f2.writelines(candidate_line)
               
            f1.close()
            fo.close()
            f2.close()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, metavar='PATH')
    parser.add_argument('--candidate_file', required=True, metavar='PATH')
    parser.add_argument('--output_file', required=True, metavar='PATH')
    args = parser.parse_args()
    main(args)

    

    
