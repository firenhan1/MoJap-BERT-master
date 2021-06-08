# ModernBERT

Sentence scoring with BERT.

This BERT implementation has the following modifications:

- Input contains only a single sentence.
- Discard the NSP task and only train on MLM objective.
- [MASK] token is replaced 100% of the time instead of 80% in the original BERT.
- Remove special tokens [CLS] and [SEP].



# Installation (pytorch=1.6.0 , torchvision=0.7.0, python=3.7.6, transformers=4.0.1, fugashi=1.0.6, ipadic=1.0.0, datasets=1.1.3)
pip install torch==1.6.0 torchvision==0.7.0 
pip install transformers==4.0.1    
pip install fugashi==1.0.6 ipadic==1.0.0
pip install datasets==1.1.3
pip install scikit-learn==0.20.3
pip install python==3.7.6


# Generate dictionary from dataset
python gen_dict.py \
    --input_file="CandidatesOCR/*.txt" \
    --output_file="vocab.txt" \
    --encoding="utf-8"



# train from a scratch

python new_train_japanese.py \
--max_position_embeddings=32 \
--do_train \
--train_data_files="train data/*.txt" \
--output_dir="output/" \
--per_device_train_batch_size=64 \
--per_device_eval_batch_size=64 \
--num_train_epochs=3 \
--load_best_model_at_end \
--save_steps=50000 \
--logging_dir="logging/" \
--logging_steps=50000 \
--save_total_limit=2 \
--seed=12345 \

---------------------------------------------------------------------------

# train from a scratch and evaluate during training (set evaluation_strategy = "steps" or "epoch")

python new_train_japanese.py \
--max_position_embeddings=32 \
--do_train \
--train_data_files="train data/*.txt" \
--output_dir="output/" \
--do_eval \
--eval_data_files="eval data/*.txt" \
--evaluation_strategy="steps" \
--per_device_train_batch_size=64 \
--per_device_eval_batch_size=64 \
--num_train_epochs=3 \
--load_best_model_at_end \
--save_steps=50000 \
--logging_dir="logging/" \
--logging_steps=50000 \
--save_total_limit=2 \
--seed=12345 \

----------------------------------------------------------------------------------

# continue to train from checkpoint and evaluate during training (load the last checkpoint, example: --model_name_or_path="output/checkpoint-300000")

python new_train_japanese.py \
--max_position_embeddings=32 \
--model_name_or_path="output/checkpoint-300000" \
--cache_dir="output/checkpoint-300000" \
--do_train \
--train_data_files="train data/*.txt" \
--output_dir="output/" \
--do_eval \
--eval_data_files="eval data/*.txt" \
--evaluation_strategy="steps" \
--per_device_train_batch_size=64 \
--per_device_eval_batch_size=64 \
--num_train_epochs=3 \
--load_best_model_at_end \
--save_steps=50000 \
--logging_dir="logging/" \
--logging_steps=50000 \
--save_total_limit=2 \
--seed=12345 \

-------------------------------------------------------------------------------------

# only evaluate existed model (load the model, example: --model_name_or_path="output/")

python new_train_japanese.py \
--max_position_embeddings=32 \
--model_name_or_path="output/" \
--cache_dir="output/" \
--output_dir="output/" \
--do_eval \
--eval_data_files="eval data/*.txt" \
--per_device_train_batch_size=64 \
--per_device_eval_batch_size=64 \
--num_train_epochs=3 \
--load_best_model_at_end \
--save_steps=50000 \
--logging_dir="logging/" \
--logging_steps=50000 \
--save_total_limit=2 \
--seed=12345 \

-------------------------------------------------------------------------------------

# rescore n-candidates OCR by BERT (only use with files which has OCR score or has formater as test_decode_result_30k.txt)

python get_score.py \
--input_file="CandidatesOCR/train_decode_result_30k.txt" \
--model_path="output/" \
--output_dir="CandidatesOCR-rescore/test_rescore" \
--model_max_length=32



-------------------------------------------------------------------------------------

# rescore n-candidates OCR by LSTM (only use with files which has OCR score or has formater as test_decode_result_30k.txt)

python get_score_lstm.py \
--input_file="CandidatesOCR/test_decode_result_30k.txt" \
--model_path="model/CharLM_data_left2right_left2right.h5" \
--mapping_file="model/data_char_mapping.pkl" \
--output_dir="CandidatesOCR-rescore-lstm/test_rescore/" \
--seq_length=20




# Find Candidate example
python find_candidates.py \
--input_file="CandidatesOCR/caption_train.txt" \
--candidate_file="CandidatesOCR/train_decode_result_30k.txt" \
--output_file="CandidatesOCR/output1.txt"