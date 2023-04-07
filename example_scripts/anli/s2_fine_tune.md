# Description
The goal of this set of experiments is to 
1) run the baseline of using RAG framework
2) run the baseline of no using retrieval, but use FID only

# Setup
## Download passage files
```
python3 .\new_atlas\preprocessing\download_corpus.py --corpus=corpora/wiki/enwiki-dec2018 --output_directory=drive/MyDrive/data/corpora
```

# RAG

We first train a simple/small model. 
## Fine-tune few short models
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
                --train_data=drive/MyDrive/data/new_atlas/anli_data/train_10.jsonl \ 
                --eval_data=drive/MyDrive/data/new_atlas/anli_data/dev_10.jsonl \ 
                --write_results \
                --generation_max_length=2 \
                --reader_model_type=google/flan-t5-small \
                --text_maxlength=512 \
                --target_maxlength=5 \
                --n_context=1 \
                --total_steps=4 \
                --save_freq=3 \
                --retriever_n_context=1 \
                --checkpoint_dir=drive/MyDrive/experiments/rag/new_atlas/results \
                --index_mode=flat \
                --task=qa \
                --passages='drive/MyDrive/data/corpora/wiki/enwiki-dec2018/text-list-100-sec.200.jsonl drive/MyDrive/data/corpora/wiki/enwiki-dec2018/infobox.200.jsonl' \ 
                --save_index_n_shards=1 \
                --num_encoder_layers=1 \
                --num_decoder_layers=1 \
                --num_retriever_layers=1 \
                --retriever_hidden_size=16 \
                --ignore_mismatched_sizes=True \
                --num_retriever_attention_heads=2 \
```

## Larger training size
We then increase the model size and observe its eval loss. 

```
export CUDA_VISIBLE_DEVICES=0
sh new_atlas/example_scripts/anli/s2_fine_tune.sh 16 2 2 2 10000

export CUDA_VISIBLE_DEVICES=1
sh new_atlas/example_scripts/anli/s2_fine_tune.sh 64 2 2 2 10000

export CUDA_VISIBLE_DEVICES=2
sh new_atlas/example_scripts/anli/s2_fine_tune.sh 64 2 6 6 10000

export CUDA_VISIBLE_DEVICES=3
sh new_atlas/example_scripts/anli/s2_fine_tune.sh 128 8 12 12 100000

export CUDA_VISIBLE_DEVICES=1
sh new_atlas/example_scripts/anli/s2_fine_tune.sh 128 4 12 12 100000

export CUDA_VISIBLE_DEVICES=0
sh new_atlas/example_scripts/anli/s2_fine_tune.sh 64 4 12 12 100000

```

### Results

        |   EM    | F1  | Eval loss (10k) | Eval loss (100k)
----    |  -----  | --- | -----     | ---- |
16/2/2/2|  0      | 0   | 0.444     | 
64/2/2/2|  0      | 0   | 0.439     |
64/2/6/6|  0      | 0   | 0.424     |
64/2/6/8|  0      | 0   | 0.416     |
64/4/12/12  |     |     |           |
128/4/12/12 |     |     |           |
128/8/12/12 |     |     |           |

The above table shows that the larger the model, the smaller the loss. 

# FID only
We remove the retrieval model and the retrieval results. This want to check the eval loss with only FID model.

```
export CUDA_VISIBLE_DEVICES=0
sh new_atlas/example_scripts/anli/s2_fine_tune.no_retrieval.sh 12 12 10000
```