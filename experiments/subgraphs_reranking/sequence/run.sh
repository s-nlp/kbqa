# python3 train_ranking_model.py 1may_Mixtral_hl_determ \
#     --output_path /mnt/storage/QA_System_Project/subgraphs_reranking_runs/determ/Mixtral/ \
#     --do_highlighting True \
#     --sequence_type determ \
#     --wandb_on True \
#     --data_config mixtral_subgraphs

python3 train_ranking_model.py 1may_Mixtral_nohl_determ \
    --output_path /mnt/storage/QA_System_Project/subgraphs_reranking_runs/determ/Mixtral/ \
    --do_highlighting False \
    --sequence_type determ \
    --wandb_on True \
    --data_config mixtral_subgraphs

python3 train_ranking_model.py 1may_Mixtral_question_answer \
    --output_path /mnt/storage/QA_System_Project/subgraphs_reranking_runs/determ/Mixtral/ \
    --sequence_type question_answer \
    --wandb_on True \
    --data_config mixtral_subgraphs

python3 train_ranking_model.py 1may_Mixtral_hl_t5 \
    --output_path /mnt/storage/QA_System_Project/subgraphs_reranking_runs/determ/Mixtral/ \
    --do_highlighting True \
    --sequence_type t5 \
    --wandb_on True \
    --data_config mixtral_subgraphs

python3 train_ranking_model.py 1may_Mixtral_nohl_t5 \
    --output_path /mnt/storage/QA_System_Project/subgraphs_reranking_runs/determ/Mixtral/ \
    --do_highlighting False \
    --sequence_type t5 \
    --wandb_on True \
    --data_config mixtral_subgraphs

python3 train_ranking_model.py 1may_Mixtral_hl_gap \
    --output_path /mnt/storage/QA_System_Project/subgraphs_reranking_runs/determ/Mixtral/ \
    --do_highlighting True \
    --sequence_type gap \
    --wandb_on True \
    --data_config mixtral_subgraphs

python3 train_ranking_model.py 1may_Mixtral_nohl_gap \
    --output_path /mnt/storage/QA_System_Project/subgraphs_reranking_runs/determ/Mixtral/ \
    --do_highlighting False \
    --sequence_type gap \
    --wandb_on True \
    --data_config mixtral_subgraphs