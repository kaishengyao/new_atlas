# Download passage files
```
python3 .\new_atlas\preprocessing\download_corpus.py --corpus=corpora/wiki/enwiki-dec2018 --output_directory=data/corpora
```


# Fine-tune few short models
```
python3 new_atlas/train.py --shuffle --train_retriever \
                --gold_score_mode=pdist \
                --use_gradient_checkpoint_reader \
                --use_gradient_checkpoint_retriever \
                --precision=fp32 \
                --temperature_gold=0.01 \
                --temperature_score=0.01 \
                --refresh_index=-1 \
                --query_side_retriever_training \
                --name=my-anli-2index-train \
                --per_gpu_batch_size=1 \
                --per_gpu_embedder_batch_size=1 \
                --max_number_passages=10 \
                --train_data=data/new_atlas/anli_data/train_10.jsonl \ 
                --eval_data=data/new_atlas/anli_data/dev_10.jsonl \ 
                --write_results \
                --generation_max_length=2 \
                --reader_model_type=google/flan-t5-small \
                --text_maxlength=512 \
                --target_maxlength=5 \
                --n_context=1 \
                --total_steps=4 \
                --save_freq=3 \
                --retriever_n_context=1 \
                --checkpoint_dir=experiments/rag/new_atlas/results \
                --index_mode=flat \
                --task=qa \
                --passages='data/corpora/wiki/enwiki-dec2018/text-list-100-sec.200.jsonl data/corpora/wiki/enwiki-dec2018/infobox.200.jsonl' \ 
                --save_index_n_shards=1 \
                --num_encoder_layers=1 \
                --num_decoder_layers=1 \
                --num_retriever_layers=1 \
                --retriever_hidden_size=16 \
                --ignore_mismatched_sizes=True \
                --num_retriever_attention_heads=2 \
```
