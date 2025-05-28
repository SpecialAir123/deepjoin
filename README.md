<div>
    <h1>DeepJoin</h1>  DeepJoin: Joinable Table Discovery with Pre-trained Language
Models

</div>


<br>

<h2>Folder Structure</h2>

```
.
├─── deepjoin_train.py             # Pretrain
├─── deepjoin_infer.py                # Index the columns of tables   
└─── deepjoin.md
```

<br>

<h2>Training Steps</h2>

**Step1: Check your environment**

You need to properly install nvidia driver first. To use GPU in a docker container You also need to install nvidia-docker2 ([Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)). Then, Please check your CUDA version via `nvidia-smi`

**Step2: Pretrain**

Here are some parameters:

> -- dataset [choose task, str] [opendata, webtable]
> --opendata [train_model name str] [all-mpnet-base-v2]
> --model_save_path [trained_model_save_path,str]
> --file_train_path  [pretain_file_path,str]
> --train_csv_file [pretrain_file path str]
> --storepath [pretrain index store path str]

```sh
python deepjoin_train.py --dataset opendata --opendata all-mpnet-base-v2 --model_save_path /deepjoin/model/output  
```

**Step3: infer**

Here are some parameters:

> -- dataset [choose task, str] [opendata, webtable]
> --datafile [infer_tables_file ,str]
> --storepath [final_reslut_storepath,str]
```sh
python deepjoin_infer.py 
```



**Step4: Full End-to-End Inference**

Here are the parameters for running the complete pipeline (`deepjoin_run_all.py`), which will:

1. Convert your query and seller CSVs into text prompts  
2. Embed them with your fine-tuned DeepJoin model  
3. Perform FAISS search to find the top-K joinable columns  

> --query_csv    [Path to your query CSV file, str]  
> --query_col    [Column name in the query CSV to join on, str]  
> --datalake_dir [Directory containing your seller CSVs, str]  
> --topk         [Number of joinable columns to retrieve, int]  
> --model_path   [Path to your fine-tuned SentenceTransformer model, str]  

```sh
python deepjoin_run_all.py \
  --query_csv    data/query.csv \
  --query_col    Country \
  --datalake_dir data/sellers/ \
  --topk         10 \
  --model_path   output/deepjoin_webtable_training-all-mpnet-base-v2-2023-10-18_19-54-27







