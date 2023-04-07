encoder_layers=$1
decoder_layers=$2
total_steps=$3

python3 new_atlas/train.py --shuffle --use_file_passages \
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
                --total_steps=${total_steps} \
                --save_freq=5000 \
                --retriever_n_context=10 \
                --checkpoint_dir=drive/MyDrive/experiments/rag/new_atlas/results \
                --index_mode=flat \
                --task=qa \
                --save_index_n_shards=2 \
                --ignore_mismatched_sizes=True \
                --num_encoder_layers=${encoder_layers} \
                --num_decoder_layers=${decoder_layers} \
                --ignore_mismatched_sizes=True 