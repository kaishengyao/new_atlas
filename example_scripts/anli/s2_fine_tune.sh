retriever_hidden_size=$1
encoder_layers=$2
decoder_layers=$3
retriever_layers=$4

python3 new_atlas/train.py --shuffle --train_retriever \
                --gold_score_mode=pdist \
                --use_gradient_checkpoint_reader \
                --use_gradient_checkpoint_retriever \
                --precision=bf16 \
                --temperature_gold=0.01 \
                --temperature_score=0.01 \
                --refresh_index=-1 \
                --query_side_retriever_training \
                --name=my-anli-2index-train \
                --per_gpu_batch_size=16 \
                --per_gpu_embedder_batch_size=16 \
                --train_data=drive/MyDrive/data/new_atlas/anli_data/train.jsonl \
                --eval_data=drive/MyDrive/data/new_atlas/anli_data/dev.jsonl \
                --write_results \
                --generation_max_length=2 \
                --reader_model_type=google/flan-t5-small \
                --text_maxlength=512 \
                --target_maxlength=5 \
                --n_context=10 \
                --total_steps=10000 \
                --save_freq=500 \
                --retriever_n_context=10 \
                --checkpoint_dir=drive/MyDrive/experiments/rag/new_atlas/results \
                --index_mode=flat \
                --task=qa \
                --passages='drive/MyDrive/data/corpora/wiki/enwiki-dec2018/text-list-100-sec.jsonl drive/MyDrive/data/corpora/wiki/enwiki-dec2018/infobox.jsonl' \
                --save_index_n_shards=2 \
                --ignore_mismatched_sizes=True \
                --num_retriever_attention_heads=2 \
                --retriever_hidden_size=${retriever_hidden_size} \
                --num_encoder_layers=${encoder_layers} \
                --num_decoder_layers=${decoder_layers} \
                --num_retriever_layers=${retriever_layers} \
                --ignore_mismatched_sizes=True 