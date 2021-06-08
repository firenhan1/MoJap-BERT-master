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

    def get_score(input, tokenizer, model, max_length):
        # encodings = tokenizer.encode_plus(input, return_tensors='pt', add_special_tokens = False, truncation=True, padding = True, max_length=max_length)
        # stride = 1
        # lls = []
        # for i in range(0, encodings.input_ids.size(1), stride):
        #     begin_loc = max(i + stride - max_length, 0)
        #     end_loc = min(i + stride, encodings.input_ids.size(1))
        #     trg_len = end_loc - i    # may be different from stride on last loop
        #     input_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
        #     target_ids = input_ids.clone()
        #     target_ids[:,:-trg_len] = -100
        #     with torch.no_grad():
        #         outputs = model(input_ids, labels=target_ids)
        #         log_likelihood = outputs[0] * trg_len
        #     lls.append(log_likelihood)
        # ppl = torch.exp(torch.stack(lls).sum() / end_loc)
        # return ppl.item()


    
        tokenize_input = tokenizer.tokenize(input)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).to(device)
        sen_len = len(tokenize_input)
        sum_log_likelihood = []
        sentence_loss = 0
        for i, word in enumerate(tokenize_input):
            # add mask to i-th character of the sentence
            tokenize_input[i] = '[MASK]'
            mask_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)]).to(device)
            output = model(mask_input, labels=tensor_input)

            # prediction_scores = output[0]
            # softmax = torch.nn.Softmax(dim=0)

            # ps = softmax(output[1][0, i]).log()
            # word_loss = ps[tensor_input[0, i]]
            # sentence_loss += word_loss.item()

            sum_log_likelihood.append(output[0])

            tokenize_input[i] = word

        stack = torch.stack(sum_log_likelihood)
        score = stack.sum().item()
        return score

    def get_score_for_long_sentence(long_sentence, tokenizer, model, max_length):
        result = 0
        if (len(long_sentence) <= max_length):
            result = get_score(long_sentence, tokenizer, model, max_length)
        else:
            chunk_sentence = []
            start = 0
            sen_length = len(long_sentence)
            while sen_length - start >= max_length:
                chunk_sentence.append(long_sentence[start:start+max_length])
                start += max_length
            chunk_sentence.append(long_sentence[start:])
            for sentence in chunk_sentence:
                score = get_score(sentence, tokenizer, model, max_length)
                result += score
        
        return result

    # args = parser.parse_args()
    tokenizer = BertJapaneseTokenizer.from_pretrained(args.model_path, model_max_length=int(args.model_max_length))
    bertMaskedLM = BertForMaskedLM.from_pretrained(args.model_path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    bertMaskedLM.to(device)

    print(glob(args.input_file))
    for file in glob(args.input_file):
        newList = []
        fileName = os.path.basename(file)
        baseName = os.path.splitext(fileName)[0]
        extension = os.path.splitext(fileName)[1]
        lamda_value_list = np.arange(0.1, 1.0, 0.1)
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
                        score_lm = get_score_for_long_sentence(sentence, tokenizer, bertMaskedLM, int(args.model_max_length))
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
    parser.add_argument('--output_dir', required=True, metavar='PATH')
    parser.add_argument('--model_max_length', required=True)
    args = parser.parse_args()
    main(args)

    

    
