# Running AMD LLama2-70B LoRA PyTorch MLPerf Benchmark (Primus)

This benchmark represents a LLama2-70B LoRA finetuning on the [GovReport](https://gov-report-data.github.io/) dataset.

## Prerequisites

- Access to the `meta-llama/Llama-2-70b-hf` model on HuggingFace
(requires signing the LLAMA 2 COMMUNITY LICENSE AGREEMENT)
- A HuggingFace access token (`HF_TOKEN`)
- ~270 GB free disk space for model download + conversion

## Step 0: Setup the host
This repo is structured around running with apptainer. Install apptainer 1.5.4 or newer.

Edit run_with_apptainer.sh with the directory paths you wish to use, data must have > 270GB and the directory where the model is stored must have 40GB or more.

CODE is the path to this repository

## Step 1: Prepare the image

Run the following build command from the root of the repository. The build process will take a while to complete. Ensure that all scripts have write access by running `sudo chmod -R 777'.

```bash
MODEL_DIR=/home/$USER/ # feel free to change this to what you set in the run_with_apptainer  script.
pushd .
cd $MODEL_DIR
apptainer pull docker://rocm/amd-mlperf:llama2_70b_training_6.0 amd-mlperf_llama2_70b_training_6.0.sif
```
Apply the patches to the image
```bash
# assumes you are in this repository
apptainer build --fakeroot --force $MODEL_DIR/amd-mlperf_llama2_70b_training_6.0_mi300x.sif mi300x.def # edit the .def file to point $MODEL_DIR
```

## Step 2: Prepare Dataset

## General Information

GovReport is a dataset for long document summarization that consists of reports written by government research agencies. The dataset hosted on the MLPerf drive is already tokenized and packed so that each sequence has length 8192.

The used model is the LLama2-70B with fused QKV. You will need 270GB to download and convert the model.

## Download and Preprocess Data & Model

To download the model from Huggingface, you'll need to sign the LLAMA 2 COMMUNITY LICENSE AGREEMENT as well as obtain a Hugginface Token `HF_TOKEN`.

Start the docker container by mounting the volume you want to use for downloading the data under `/data` within the container. In this example we use `/data/mlperf_llama2/data` as the host download directory:

```bash
apptainer run --rocm amd-mlperf_llama2_70b_training_6.0_mi300x.sif
```

Start the script for downloading and preprocessing data from within the container (HF_TOKEN is needed to download the model weights):

```bash
export HF_TOKEN=<your-huggingace-token>
bash ./scripts/prepare_data_and_model.sh
```

## Exit Container

Exit the container by running the below command

```bash
exit
```

## Verify Data

After completion of the above step, your data directory, on the host, should look like this:

```
/data/mlperf_llama2/data
├── train.npy                          # Packed training data (seq_length=8192)
├── validation.npy                     # Packed validation data
├── packed_metadata.jsonl              # Sequence metadata for Megatron-Bridge
├── megatron_checkpoints/
│   └── Llama-2-70b-hf/
│       ├── latest_checkpointed_iteration.txt
│       ├── latest_train_state.pt
│       └── iter_0000000/
│           ├── __0_0.distcp           # Sharded weights (~64 GB)
│           ├── __0_1.distcp           # Sharded weights (~64 GB)
│           ├── common.pt
│           ├── .metadata
│           ├── metadata.json
│           ├── run_config.yaml
│           ├── modelopt_run_config.yaml
│           ├── train_state.pt
│           └── tokenizer/
│               ├── tokenizer.model
│               ├── tokenizer_config.json
│               └── special_tokens_map.json

```

# 3. Run Training


Load the config filie
```bash
source config_MI300X_1x8x1.sh 
```

### Launch 1 Training Run

If you want to perform a single run, use:

```bash
export HF_TOKEN=<your-huggingace-token>
export NEXP=1
bash run_with_docker.sh
```

### Launch 10 Training Run [Optional]

If you would like to prepare for 10 run submisision, use:

```bash
export HF_TOKEN=<your-huggingace-token>
# set nexp in config_MI300X
bash run_with_apptainer.sh
```

After completion, the logs will be available under the directory `$LOGDIR`.

Note:To optimize the machine's performance, the training script will also execute `runtime_tunables.sh` script before any training run.

# 4. Check Quality

### Quality Metric

Cross entropy loss

### Quality Target

0.925
