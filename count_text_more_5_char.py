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

    with open(args.input_file, encoding='utf-8') as fo:
        result = [s for s in fo]
        real_count = 0
        count_line_more_5_char = 0
        for line in result:
            result_line = line.rstrip()
            result_elements = result_line.split()
            result_title = result_elements[0]
            if line != '':
                real_count = real_count + 1
            if len(result_elements) > 1:
                result_sentence = result_elements[1:len(result_elements)]
                if len(result_sentence) >= 5:
                    count_line_more_5_char = count_line_more_5_char + 1

        print('real text lines number: ', real_count)
        print('text lines number more than 5 characters: ', count_line_more_5_char)       
        fo.close()



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, metavar='PATH')
    args = parser.parse_args()
    main(args)

    

    
