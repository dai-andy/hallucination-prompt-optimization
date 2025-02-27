python run.py --seed 5 --dataset hallucination --task cls --batch-size 4 --prompt-num 0 --sample_num 4 --language_model gpt --budget 4 --popsize 3 --position demon --evo_mode ga --llm_type turbo --setting default --initial all --initial_mode topk --ga_mode topk --cache_path data/cls/hallucination/seed5/prompts_batched.json --output outputs/cls/hallucination/turbo/all/ga/bd5_top10_para_topk_init/topk/turbo/seed5 --dev_file ./data/cls/hallucination/seed5/dev.txt --sel_mode tour

# !/bin/bash

# set -ex

# # Set Python path and environment variables
# export PYTHONPATH=${PYTHONPATH}:$(pwd)
# export CUBLAS_WORKSPACE_CONFIG=:16:8  
# export CUDA_VISIBLE_DEVICES=0

# # Configuration
# BUDGET=5
# POPSIZE=10
# SEED=5
# GA=topk
# LLM_TYPE=turbo

# for dataset in hallucination
# do
#     OUT_PATH=outputs/cls/${dataset}/turbo/all/ga/bd${BUDGET}_top${POPSIZE}_para_topk_init/${GA}/${LLM_TYPE}
    
#     # Create output directory if it doesn't exist
#     mkdir -p ${OUT_PATH}
    
#     for SEED in 5
#     do
#         python3 run.py \
#             --seed ${SEED} \
#             --dataset ${dataset} \
#             --task cls \
#             --batch-size 32 \
#             --prompt-num 0 \
#             --sample_num 500 \
#             --language_model gpt \
#             --budget ${BUDGET} \
#             --popsize ${POPSIZE} \
#             --position demon \
#             --evo_mode ga \
#             --llm_type ${LLM_TYPE} \
#             --setting default \
#             --initial all \
#             --initial_mode para_topk \
#             --ga_mode ${GA} \
#             --cache_path data/cls/${dataset}/seed${SEED}/prompts_batched.json \
#             --output ${OUT_PATH}/seed${SEED} \
#             --dev_file ./data/cls/${dataset}/seed${SEED}/dev.txt
#     done
    
#     python3 get_result.py -p ${OUT_PATH} > ${OUT_PATH}/result.txt
# done