run_id=$1
bsz=128

CUDA_VISIBLE_DEVICES=0 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:1750 --nnodes=1 --nproc_per_node=1 inter_res_bank_new.py \
	--data-path /home/cxk/cxk/workplace/VisualSearch/visdial_1.0_train \
	--amp --run_id $run_id --batch-size $bsz --model_name $model_name --backbone_ft --test_model 

	
