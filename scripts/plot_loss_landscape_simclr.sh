
MODEL_CKPT_SIMCLR0=SupContrast/save/SimCLR/cifar10_models/cifar10_resnet18_lr_0.05_decay_0.0001_bsz_256_temp_0.07_cosine_seed_0/model_200.pth
MODEL_CKPT_SIMCLR10=SupContrast/save/SimCLR/cifar10_models/cifar10_resnet18_lr_0.05_decay_0.0001_bsz_256_temp_0.07_cosine_seed_10/model_200.pth
MODEL_CKPT_SIMCLR20=SupContrast/save/SimCLR/cifar10_models/cifar10_resnet18_lr_0.05_decay_0.0001_bsz_256_temp_0.07_cosine_seed_20/model_200.pth
MODEL_CKPT_SIMCLR30=SupContrast/save/SimCLR/cifar10_models/cifar10_resnet18_lr_0.05_decay_0.0001_bsz_256_temp_0.07_cosine_seed_30/model_200.pth

CLF_CKPT_SIMCLR0=SupContrast/save/linear_tuning/SimCLR/cifar100/simclr_on_cifar10_seed_0
CLF_CKPT_SIMCLR10=SupContrast/save/linear_tuning/SimCLR/cifar100/simclr_on_cifar10_seed_10
CLF_CKPT_SIMCLR20=SupContrast/save/linear_tuning/SimCLR/cifar100/simclr_on_cifar10_seed_20
CLF_CKPT_SIMCLR30=SupContrast/save/linear_tuning/SimCLR/cifar100/simclr_on_cifar10_seed_30

tasks=("[0,1,2,3,4]" "[95,96,97,98,99]" "[6,11,16,21,26]" "[56,58,62,66,68]")
task_names=("task_0_1_2_3_4" "task_95_96_97_98_99" "task_6_11_16_21_26" "task_56_58_62_66_68")

TASK_INDEX=0
for task in "${tasks[@]}"
do
    task_name=${task_names[$TASK_INDEX]}

    # append the task_name to the clf checkpoint path
    CLF_CKPT_SIMCLR0_TASK=$CLF_CKPT_SIMCLR0/$task_name/models/classifier_30.pth
    CLF_CKPT_SIMCLR10_TASK=$CLF_CKPT_SIMCLR10/$task_name/models/classifier_30.pth
    CLF_CKPT_SIMCLR20_TASK=$CLF_CKPT_SIMCLR20/$task_name/models/classifier_30.pth
    CLF_CKPT_SIMCLR30_TASK=$CLF_CKPT_SIMCLR30/$task_name/models/classifier_30.pth

    python3 plot_surface.py task=cifar100 task.labs=$task +map=simclr plot_loss.model_ckpt_name=$MODEL_CKPT_SIMCLR0 plot_loss.clf_ckpt_name=$CLF_CKPT_SIMCLR0_TASK plot_loss.pretrained_tag=simclr_on_cifar10 plot_loss.seed=0
    python3 plot_surface.py task=cifar100 task.labs=$task +map=simclr plot_loss.model_ckpt_name=$MODEL_CKPT_SIMCLR10 plot_loss.clf_ckpt_name=$CLF_CKPT_SIMCLR10_TASK plot_loss.pretrained_tag=simclr_on_cifar10 plot_loss.seed=10
    python3 plot_surface.py task=cifar100 task.labs=$task +map=simclr plot_loss.model_ckpt_name=$MODEL_CKPT_SIMCLR20 plot_loss.clf_ckpt_name=$CLF_CKPT_SIMCLR20_TASK plot_loss.pretrained_tag=simclr_on_cifar10 plot_loss.seed=20
    python3 plot_surface.py task=cifar100 task.labs=$task +map=simclr plot_loss.model_ckpt_name=$MODEL_CKPT_SIMCLR30 plot_loss.clf_ckpt_name=$CLF_CKPT_SIMCLR30_TASK plot_loss.pretrained_tag=simclr_on_cifar10 plot_loss.seed=30

    let TASK_INDEX=${TASK_INDEX}+1

done