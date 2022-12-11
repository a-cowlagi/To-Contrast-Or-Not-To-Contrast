

MODEL_CKPT_SUPCON0=SupContrast/save/SupCon/cifar10_models/cifar10_resnet18_lr_0.05_decay_0.0001_bsz_256_temp_0.07_cosine_seed_0/model_200.pth
MODEL_CKPT_SUPCON10=SupContrast/save/SupCon/cifar10_models/cifar10_resnet18_lr_0.05_decay_0.0001_bsz_256_temp_0.07_cosine_seed_10/model_200.pth
MODEL_CKPT_SUPCON20=SupContrast/save/SupCon/cifar10_models/cifar10_resnet18_lr_0.05_decay_0.0001_bsz_256_temp_0.07_cosine_seed_20/model_200.pth
MODEL_CKPT_SUPCON30=SupContrast/save/SupCon/cifar10_models/cifar10_resnet18_lr_0.05_decay_0.0001_bsz_256_temp_0.07_cosine_seed_30/model_200.pth

CLF_CKPT_SUPCON0=SupContrast/save/linear_tuning/SupCon/cifar100/supcon_on_cifar10_seed_0
CLF_CKPT_SUPCON10=SupContrast/save/linear_tuning/SupCon/cifar100/supcon_on_cifar10_seed_10
CLF_CKPT_SUPCON20=SupContrast/save/linear_tuning/SupCon/cifar100/supcon_on_cifar10_seed_20
CLF_CKPT_SUPCON30=SupContrast/save/linear_tuning/SupCon/cifar100/supcon_on_cifar10_seed_30

tasks=("[0,1,2,3,4]" "[95,96,97,98,99]" "[6,11,16,21,26]" "[56,58,62,66,68]")
task_names=("task_0_1_2_3_4" "task_95_96_97_98_99" "task_6_11_16_21_26" "task_56_58_62_66_68")

TASK_INDEX=0
for task in "${tasks[@]}"
do
    task_name=${task_names[$TASK_INDEX]}

    # append the task_name to the clf checkpoint path
    CLF_CKPT_SUPCON0_TASK=$CLF_CKPT_SUPCON0/$task_name/models
    CLF_CKPT_SUPCON10_TASK=$CLF_CKPT_SUPCON10/$task_name/models
    CLF_CKPT_SUPCON20_TASK=$CLF_CKPT_SUPCON20/$task_name/models
    CLF_CKPT_SUPCON30_TASK=$CLF_CKPT_SUPCON30/$task_name/models

    python3 map_finetuning_logits.py task=cifar100 task.labs=$task +map=supcon map.model_ckpt_name=$MODEL_CKPT_SUPCON0 +map.clf_ckpt_dir=$CLF_CKPT_SUPCON0_TASK +map.seed=0
    python3 map_finetuning_logits.py task=cifar100 task.labs=$task +map=supcon map.model_ckpt_name=$MODEL_CKPT_SUPCON10 +map.clf_ckpt_dir=$CLF_CKPT_SUPCON10_TASK +map.seed=10
    python3 map_finetuning_logits.py task=cifar100 task.labs=$task +map=supcon map.model_ckpt_name=$MODEL_CKPT_SUPCON20 +map.clf_ckpt_dir=$CLF_CKPT_SUPCON20_TASK +map.seed=20
    python3 map_finetuning_logits.py task=cifar100 task.labs=$task +map=supcon map.model_ckpt_name=$MODEL_CKPT_SUPCON30 +map.clf_ckpt_dir=$CLF_CKPT_SUPCON30_TASK +map.seed=30

    let INDEX=${INDEX}+1
done