CUDA_VISIBLE_DEVICES=1,2 \
	torchrun --nproc_per_node=2 train.py	\
	--add_batch \
	--long_semantic \
	--para_file ../../data/instruction_para.json

