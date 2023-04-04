# Experiments

## Base model

```
python3 new_atlas/preprocessing/download_model.py --model models/atlas/base --output_directory drive/MyDrive/experiments/rag/new_atlas/models/base
```

Models are saved under 
```
.\drive/MyDrive/experiments\rag\new_atlas\models\base\models\atlas\base\
```

Size: 713101951 model.pth.tar



## Index
```
python3 new_atlas/preprocessing/download_index.py --index indices/atlas/wiki/base --output_directory drive/MyDrive/experiments/rag/new_atlas/index/base
```


# Prepare ANLI data
```
python3 new_atlas/preprocessing/prepare_anli.py --output_directory=drive/MyDrive/data/new_atlas
```

The number of samples is in the following table.
       | Train | Dev | Test |
   --- | ----  | --- | ---- | 
Counts | 100459| 1200 | 1200 |


# Evaluate
Since the index is large with more 20 shards. To save computing, I only use 2 shards for experimentations. 

```
size=base

python3 new_atlas/evaluate.py --name=my-anli-2index-evaluation-test10 \
    --generation_max_length=2 \
    --gold_score_mode=pdist \
    --precision=fp32 \
    --reader_model_type=google/t5-base-lm-adapt \
    --text_maxlength=512 \
    --target_maxlength=5 \
    --model_path=drive/MyDrive/experiments/rag/new_atlas/models/base/models/atlas/base \
    --eval_data=drive/MyDrive/data/new_atlas/anli_data/test.jsonl \
    --per_gpu_batch_size=1 \
    --n_context=40 --retriever_n_context=40 \
    --checkpoint_dir=drive/MyDrive/experiments/rag/new_atlas/results \
    --index_mode=flat  \
    --task=qa \
    --load_index_path=drive/MyDrive/experiments/rag/new_atlas/index/base/indices/atlas/wiki/base \
    --write_results \
    --save_index_n_shards=2
```

# TODO

1. Support precision fp16 for retrieval score 