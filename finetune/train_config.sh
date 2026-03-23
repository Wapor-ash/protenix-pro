#!/bin/bash
# Unified training config for finetune/train_pipeline.sh
#
# Usage:
#   1. Copy this file if you want multiple experiment presets.
#   2. Edit variables below.
#   3. Run:
#        bash finetune/train_pipeline.sh finetune/train_config.sh
#        bash finetune/inference_pipeline.sh finetune/train_config.sh
#
# Notes:
# - This config is sourced by bash, so keep values shell-safe.
# - Use "true" / "false" strings for boolean-like options.
# - Comments below list supported choices for each variable.

# ===================== Run Mode =====================
# TRAIN_MODE choices: "1stage" | "2stage"
TRAIN_MODE="1stage"

# RUN_NAME choices: any non-empty string, or "" for auto-generated name
RUN_NAME=""

# USE_WANDB choices: "true" | "false"
USE_WANDB="true"

# PRINT_ONLY choices: "true" | "false"
# When true, train_pipeline.sh / inference_pipeline.sh print the assembled command and exit.
PRINT_ONLY="false"

# CONDA_ENV choices: conda env name, absolute env path, or "" to skip activation
CONDA_ENV="/inspire/ssd/project/sais-bio/public/ash_proj/conda/envs/tune_protenix"


# ===================== Project Paths =====================
# PROJECT_ROOT choices: absolute path to your ash project root
PROJECT_ROOT="/inspire/ssd/project/sais-bio/public/ash_proj"

# PROTENIX_DIR choices: absolute path to the Protenix repo root
PROTENIX_DIR="${PROJECT_ROOT}/code/protenix_pro/Protenix"

# OUTPUT_ROOT choices: base output directory for train / inference artifacts
OUTPUT_ROOT="${PROTENIX_DIR}/output"

# DATA_DIR choices: absolute path to the RNA finetune dataset root
DATA_DIR="${PROJECT_ROOT}/data/stanford-rna-3d-folding/part2"

# PREPARED_DATA_DIR choices: absolute path to prepared Protenix data
PREPARED_DATA_DIR="${DATA_DIR}/protenix_prepared"

# CHECKPOINT_PATH choices: absolute path to a base checkpoint .pt file
CHECKPOINT_PATH="${PROTENIX_DIR}/checkpoints/protenix_base_20250630_v1.0.0.pt"

# LOAD_EMA_CHECKPOINT_PATH choices: "" or absolute path to EMA checkpoint .pt
LOAD_EMA_CHECKPOINT_PATH=""


# ===================== Dataset / Split Config =====================
# TRAIN_SET choices: Protenix dataset key string
TRAIN_SET="weightedPDB_before2109_wopb_nometalc_0925"

# VAL_SET choices: Protenix validation dataset key string
VAL_SET="recentPDB_1536_sample384_0925"

# RNA_CIF_DIR choices: absolute path to RNA mmCIF directory
RNA_CIF_DIR="${DATA_DIR}/PDB_RNA"

# BIOASSEMBLY_DIR choices: absolute path to bioassembly dict directory
BIOASSEMBLY_DIR="${PREPARED_DATA_DIR}/rna_bioassembly"

# TRAIN_INDICES_FPATH choices: absolute path to train/val indices CSV
TRAIN_INDICES_FPATH="${PREPARED_DATA_DIR}/indices/rna_bioassembly_indices.csv"

# TRAIN_PDB_LIST choices: absolute path to train pdb list txt
TRAIN_PDB_LIST="${PREPARED_DATA_DIR}/rna_train_pdb_list_filtered.txt"

# VAL_INDICES_FPATH choices: absolute path to validation indices CSV
VAL_INDICES_FPATH="${PREPARED_DATA_DIR}/indices/rna_bioassembly_indices.csv"

# VAL_PDB_LIST choices: absolute path to validation pdb list txt
VAL_PDB_LIST="${PREPARED_DATA_DIR}/rna_val_pdb_list_filtered.txt"

# VAL_FIND_EVAL_CHAIN_INTERFACE choices: "true" | "false"
VAL_FIND_EVAL_CHAIN_INTERFACE="false"

# VAL_GROUP_BY_PDB_ID choices: "true" | "false"
VAL_GROUP_BY_PDB_ID="false"

# VAL_MAX_N_TOKEN choices: positive integer
VAL_MAX_N_TOKEN="1536"


# ===================== Runtime / Environment =====================
# CUDA_VISIBLE_DEVICES_VALUE choices: e.g. "0", "0,1", or "" to keep current env
CUDA_VISIBLE_DEVICES_VALUE=""

# TRIANGLE_ATTENTION_IMPL choices: "cuequivariance" | backend supported by repo
TRIANGLE_ATTENTION_IMPL="cuequivariance"

# TRIANGLE_MULTIPLICATIVE_IMPL choices: "cuequivariance" | backend supported by repo
TRIANGLE_MULTIPLICATIVE_IMPL="cuequivariance"

# PYTHON_BIN choices: "python3" | absolute python path
PYTHON_BIN="python3"

# LAYERNORM_TYPE_VALUE choices: "" | "fast_layernorm" | "torch"
# Leave empty to keep repo default. Set "torch" if you need portable fallback.
LAYERNORM_TYPE_VALUE=""


# ===================== Common Training Params =====================
# MODEL_NAME_ARG choices: Protenix model name string
MODEL_NAME_ARG="protenix_base_20250630_v1.0.0"

# PROJECT_NAME choices: any logging project string, or "" for auto by TRAIN_MODE
PROJECT_NAME=""

# SEED choices: integer
SEED="42"

# DTYPE choices: "bf16" | "fp16" | repo-supported dtype
DTYPE="bf16"

# DIFFUSION_BATCH_SIZE choices: positive integer
DIFFUSION_BATCH_SIZE="48"

# EVAL_INTERVAL choices: positive integer
EVAL_INTERVAL="400"

# LOG_INTERVAL choices: positive integer
LOG_INTERVAL="50"

# CHECKPOINT_INTERVAL choices: positive integer
CHECKPOINT_INTERVAL="1000"

# TRAIN_CROP_SIZE choices: positive integer
TRAIN_CROP_SIZE="384"

# MAX_STEPS choices: positive integer
MAX_STEPS="80000"

# LR choices: positive float; used as global scheduler LR
LR="0.0001"

# LR_SCHEDULER choices: "af3" | repo-supported scheduler name
LR_SCHEDULER="af3"

# WARMUP_STEPS choices: non-negative integer
WARMUP_STEPS="2000"

# GRAD_CLIP_NORM choices: positive float
GRAD_CLIP_NORM="10"

# MODEL_N_CYCLE choices: positive integer
MODEL_N_CYCLE="4"

# SAMPLE_DIFFUSION_N_STEP choices: positive integer
SAMPLE_DIFFUSION_N_STEP="200"

# LOAD_STRICT choices: "true" | "false"
LOAD_STRICT="false"

# DATA_NUM_DL_WORKERS choices: non-negative integer
DATA_NUM_DL_WORKERS="0"

# EMA_DECAY choices: float in (0, 1); only used in 1stage
EMA_DECAY="0.999"

# ADAM_USE_ADAMW choices: "true" | "false"
ADAM_USE_ADAMW="true"

# ADAM_BETA1 choices: float in (0, 1)
ADAM_BETA1="0.9"

# ADAM_BETA2 choices: float in (0, 1)
ADAM_BETA2="0.999"

# ADAM_WEIGHT_DECAY choices: non-negative float
ADAM_WEIGHT_DECAY="0.01"

# LOSS_ALPHA_BOND choices: non-negative float
LOSS_ALPHA_BOND="0.5"

# RNA_LOSS_ENABLE choices: "true" | "false"
RNA_LOSS_ENABLE="false"

# CONFIDENCE_STOP_GRADIENT choices: "true" | "false"
CONFIDENCE_STOP_GRADIENT="true"


# ===================== 1-Stage Params =====================
# ADAPTER_LR choices: non-negative float
ADAPTER_LR="0.001"

# BACKBONE_LR choices: non-negative float; "0" freezes backbone in 1stage
BACKBONE_LR="0.00005"


# ===================== 2-Stage Params =====================
# STAGE1_MAX_STEPS choices: positive integer
STAGE1_MAX_STEPS="400"

# STAGE1_ADAPTER_LR choices: non-negative float
STAGE1_ADAPTER_LR="0.005"

# STAGE1_BACKBONE_LR choices: non-negative float; "0" freezes backbone in stage1
STAGE1_BACKBONE_LR="0.0"

# STAGE1_WARMUP_STEPS choices: non-negative integer
STAGE1_WARMUP_STEPS="2000"

# STAGE2_ADAPTER_LR choices: non-negative float, or "-1.0" to keep repo default semantics
STAGE2_ADAPTER_LR="-1.0"

# STAGE2_BACKBONE_LR choices: non-negative float, or "-1.0" to keep repo default semantics
STAGE2_BACKBONE_LR="-1.0"

# STAGE2_WARMUP_STEPS choices: non-negative integer
STAGE2_WARMUP_STEPS="100"

# STAGE2_EMA_DECAY choices: float in (0, 1)
STAGE2_EMA_DECAY="0.999"


# ===================== RibonanzaNet2 Params =====================
# USE_RIBONANZA choices: "true" | "false"
USE_RIBONANZA="true"

# RIBONANZA_MODEL_DIR choices: absolute path to directory containing
#   pairwise.yaml and pytorch_model_fsdp.bin
RIBONANZA_MODEL_DIR="${PROJECT_ROOT}/data/ribonanzanet2/model_weights"

# RIBONANZA_GATE_TYPE choices: "channel" | "scalar"
RIBONANZA_GATE_TYPE="channel"

# RIBONANZA_N_PAIRFORMER_BLOCKS choices: positive integer
RIBONANZA_N_PAIRFORMER_BLOCKS="4"

# ===================== RNA / DNA LLM Params =====================
# USE_RNALM choices: "true" | "false"
USE_RNALM="true"

# USE_RNA choices: "true" | "false"
USE_RNA="true"

# USE_DNA choices: "true" | "false"
USE_DNA="false"

# INJECTION_MODE choices: "input" | "diffusion" | "both"
INJECTION_MODE="diffusion"

# GATE_MODE choices: "none" | "scalar" | "token" | "dual"
GATE_MODE="none"

# GATE_INIT_LOGIT choices: float
GATE_INIT_LOGIT="-3.0"

# RNALM_MODEL_NAME choices: model tag supported by your embedding manifests, e.g. "aido"
RNALM_MODEL_NAME="aido"

# RNA_EMBEDDING_DIM choices: positive integer; current AIDO RNA is 2048
RNA_EMBEDDING_DIM="2048"

# DNA_EMBEDDING_DIM choices: positive integer; current AIDO DNA is 1024
DNA_EMBEDDING_DIM="1024"

# RNA_EMBEDDING_DIR choices: absolute path to RNA embedding dir
RNA_EMBEDDING_DIR="${DATA_DIR}/aido_embeddings/rna"

# RNA_SEQUENCE_FPATH choices: absolute path to RNA sequence CSV
RNA_SEQUENCE_FPATH="${RNA_EMBEDDING_DIR}/rna_sequences.csv"

# DNA_EMBEDDING_DIR choices: absolute path to DNA embedding dir
DNA_EMBEDDING_DIR="${DATA_DIR}/aido_embeddings/dna"

# DNA_SEQUENCE_FPATH choices: absolute path to DNA sequence CSV
DNA_SEQUENCE_FPATH="${DNA_EMBEDDING_DIR}/dna_sequences.csv"

# RNALM_SEPARATE_DNA_PROJECTION choices: "true" | "false"
RNALM_SEPARATE_DNA_PROJECTION="true"


# ===================== RNA Template Params =====================
# USE_RNA_TEMPLATE choices: "true" | "false"
USE_RNA_TEMPLATE="false"

# RNA_PROJECTOR_INIT choices: "protein" | "zero"
RNA_PROJECTOR_INIT="protein"

# RNA_TEMPLATE_ALPHA choices: non-negative float
RNA_TEMPLATE_ALPHA="0.01"

# MAX_RNA_TEMPLATES choices: positive integer
MAX_RNA_TEMPLATES="4"

# TEMPLATE_N_BLOCKS choices: non-negative integer
TEMPLATE_N_BLOCKS="2"

# MANUAL_TEMPLATE_HINTS choices: "" or absolute/relative path to training manual hints JSON
MANUAL_TEMPLATE_HINTS=""

# RNA_DATABASE_DIR choices: absolute path to RNA template database dir
RNA_DATABASE_DIR="${PROTENIX_DIR}/rna_database"

# RNA_SEARCH_RESULTS choices: absolute path to search_results.json
RNA_SEARCH_RESULTS="${RNA_DATABASE_DIR}/search_results.json"

# RNA3DB_METADATA_PATH choices: absolute path to RNA3D filter metadata JSON
RNA3DB_METADATA_PATH="${PROJECT_ROOT}/data/RNA3D/rna3db-jsons/filter.json"

# PDB_RNA_DIR choices: absolute path to RNA structure mmCIF dir
PDB_RNA_DIR="${DATA_DIR}/PDB_RNA"


# ===================== RNA SS Pair-Prior Params =====================
# USE_RNA_SS choices: "true" | "false"
USE_RNA_SS="false"

# RNA_SS_SEQUENCE_FPATH choices: "" or absolute path to sequence->prior CSV
# Empty + RNA_SS_STRICT=false means graceful zero-prior fallback.
RNA_SS_SEQUENCE_FPATH=""

# RNA_SS_FEATURE_DIR choices: "" or absolute path to prior root directory
RNA_SS_FEATURE_DIR=""

# RNA_SS_FORMAT choices: "sparse_npz" | "dense_npz"
RNA_SS_FORMAT="sparse_npz"

# RNA_SS_N_CLASSES choices: fixed to "6" for [P_in, o_i, o_j, r_i, r_j, m_ij]
RNA_SS_N_CLASSES="6"

# RNA_SS_COVERAGE_WINDOW choices: non-negative integer
RNA_SS_COVERAGE_WINDOW="8"

# RNA_SS_STRICT choices: "true" | "false"
RNA_SS_STRICT="false"

# RNA_SS_MIN_PROB choices: non-negative float
RNA_SS_MIN_PROB="0.0"

# RNA_SS_ARCHITECTURE choices: "mlp" | "transformer"
RNA_SS_ARCHITECTURE="mlp"

# RNA_SS_HIDDEN_DIM choices: positive integer
RNA_SS_HIDDEN_DIM="128"

# RNA_SS_N_LAYERS choices: positive integer
RNA_SS_N_LAYERS="3"

# RNA_SS_ALPHA_INIT choices: positive float
RNA_SS_ALPHA_INIT="0.01"

# RNA_SS_INIT_METHOD choices: "kaiming"
RNA_SS_INIT_METHOD="kaiming"


# ===================== RNA MSA / Protein Template Flags =====================
# USE_RNA_MSA choices: "true" | "false"
USE_RNA_MSA="true"

# RNA_MSA_RAW_DIR choices: absolute path to RNA MSA dir
RNA_MSA_RAW_DIR="${PREPARED_DATA_DIR}/rna_msa/msas"

# RNA_MSA_INDEX_JSON choices: absolute path to RNA msa index json
RNA_MSA_INDEX_JSON="${PREPARED_DATA_DIR}/rna_msa/rna_sequence_to_pdb_chains.json"

# RNA_MSA_INDEXING_METHOD choices: "sequence" | repo-supported indexing key
RNA_MSA_INDEXING_METHOD="sequence"

# USE_PROT_MSA choices: "true" | "false"
USE_PROT_MSA="false"

# USE_PROT_TEMPLATE choices: "true" | "false"
USE_PROT_TEMPLATE="false"


# ===================== Inference Params =====================
# INFER_CHECKPOINT_PATH choices: absolute path to inference checkpoint .pt
INFER_CHECKPOINT_PATH="${CHECKPOINT_PATH}"

# INFER_MODEL_NAME choices: logical model name used by runner/inference.py
# Usually keep it aligned with MODEL_NAME_ARG.
INFER_MODEL_NAME="${MODEL_NAME_ARG}"

# INFER_RUN_NAME choices: any string, or "" to derive from input json basename
INFER_RUN_NAME=""

# INFER_INPUT_JSON choices: absolute path to inference input JSON
INFER_INPUT_JSON=""

# INFER_DUMP_DIR choices: "" for auto under OUTPUT_ROOT/inference, or absolute path
INFER_DUMP_DIR=""

# INFER_LOAD_STRICT choices: "true" | "false"
INFER_LOAD_STRICT="false"

# INFER_NUM_WORKERS choices: non-negative integer
INFER_NUM_WORKERS="0"

# INFER_SEEDS choices: comma-separated integers, e.g. "101" or "101,102"
INFER_SEEDS="101"

# INFER_USE_SEEDS_IN_JSON choices: "true" | "false"
INFER_USE_SEEDS_IN_JSON="false"

# INFER_DTYPE choices: "bf16" | "fp16" | repo-supported dtype
INFER_DTYPE="${DTYPE}"

# INFER_N_SAMPLE choices: positive integer
INFER_N_SAMPLE="1"

# INFER_N_STEP choices: positive integer
INFER_N_STEP="${SAMPLE_DIFFUSION_N_STEP}"

# INFER_N_CYCLE choices: positive integer
INFER_N_CYCLE="${MODEL_N_CYCLE}"

# INFER_USE_MSA choices: "true" | "false"
INFER_USE_MSA="false"

# INFER_USE_TEMPLATE choices: "true" | "false"
INFER_USE_TEMPLATE="false"

# INFER_USE_RNA_MSA choices: "true" | "false"
INFER_USE_RNA_MSA="false"

# INFER_NEED_ATOM_CONFIDENCE choices: "true" | "false"
INFER_NEED_ATOM_CONFIDENCE="false"

# INFER_SORTED_BY_RANKING_SCORE choices: "true" | "false"
INFER_SORTED_BY_RANKING_SCORE="true"

# INFER_ENABLE_TF32 choices: "true" | "false"
INFER_ENABLE_TF32="true"

# INFER_ENABLE_EFFICIENT_FUSION choices: "true" | "false"
INFER_ENABLE_EFFICIENT_FUSION="true"

# INFER_ENABLE_DIFFUSION_SHARED_VARS_CACHE choices: "true" | "false"
INFER_ENABLE_DIFFUSION_SHARED_VARS_CACHE="true"


# ===================== Escape Hatches =====================
# TRAIN_EXTRA_ARGS / INFER_EXTRA_ARGS choices:
# shell-authored extra CLI overrides appended verbatim at the end.
TRAIN_EXTRA_ARGS=""
INFER_EXTRA_ARGS=""
