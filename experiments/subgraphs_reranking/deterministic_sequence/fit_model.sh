# sentence-transformers/all-mpnet-base-v2
python3 train_ranking_model.py mse_subgraph_mpnet_ranking_T5XLSSMNQ \
    hle2000/Mintaka_Subgraphs_T5_xl_ssm
    --output_path /mnt/storage/QA_System_Project/subgrraphs_reranking_runs/ \
    --model_name sentence-transformers/all-mpnet-base-v2 \
    --classification_threshold 0.5 \
    --wandb_on True \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --num_train_epochs 6

python3 train_ranking_model.py mse_subgraph_mpnet_ranking_T5LargeSSM \
    hle2000/Mintaka_Subgraphs_T5_large_ssm
    --output_path /mnt/storage/QA_System_Project/subgrraphs_reranking_runs/ \
    --model_name sentence-transformers/all-mpnet-base-v2 \
    --classification_threshold 0.5 \
    --wandb_on True \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --num_train_epochs 6

# bert-large-cased
python3 train_ranking_model.py mse_subgraph_bert_ranking_T5XLSSMNQ \
    hle2000/Mintaka_Subgraphs_T5_xl_ssm
    --output_path /mnt/storage/QA_System_Project/subgrraphs_reranking_runs/ \
    --model_name bert-large-cased \
    --classification_threshold 0.5 \
    --wandb_on True \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --num_train_epochs 6

python3 train_ranking_model.py mse_subgraph_bert_ranking_T5LargeSSM \
    hle2000/Mintaka_Subgraphs_T5_large_ssm
    --output_path /mnt/storage/QA_System_Project/subgrraphs_reranking_runs/ \
    --model_name bert-large-cased \
    --classification_threshold 0.5 \
    --wandb_on True \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --num_train_epochs 6

# Ablation study
python3 train_ranking_model.py mse_subgraph_mpnet_ranking_T5XLSSMNQ_no_highliting \
    hle2000/Mintaka_Subgraphs_T5_xl_ssm
    --output_path /mnt/storage/QA_System_Project/subgrraphs_reranking_runs/ \
    --model_name sentence-transformers/all-mpnet-base-v2 \
    --classification_threshold 0.5 \
    --wandb_on True \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --num_train_epochs 4 \
    --do_linearization True \
    --do_highlighting False
    
python3 train_ranking_model.py mse_subgraph_mpnet_ranking_T5XLSSMNQ_no_linearization \
    hle2000/Mintaka_Subgraphs_T5_xl_ssm
    --output_path /mnt/storage/QA_System_Project/subgrraphs_reranking_runs/ \
    --model_name sentence-transformers/all-mpnet-base-v2 \
    --classification_threshold 0.5 \
    --wandb_on True \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --num_train_epochs 4 \
    --do_linearization False \
    --do_highlighting False
