#Example train command
./distributed_train.sh num_proc <folder_path> --experiment <experiment_name> --img-size 224 --model MONet_T --num-classes 100 --sched cosine --epochs 300 --opt adamw --lr 0.001 --clip-grad 1 --batch-size 128 --amp 


For data augmentation, please refer to train.py to add data augmentation during training (e.g --mix_up xx --smoothing xx)
