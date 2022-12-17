### Supervised learning models

CKPT_SUPCE=SupContrast/save/SupCE/cifar10_models/cifar10_resnet18_lr_0.2_decay_0.0001_bsz_256_cosine_seed_0,SupContrast/save/SupCE/cifar10_models/cifar10_resnet18_lr_0.2_decay_0.0001_bsz_256_cosine_seed_10,SupContrast/save/SupCE/cifar10_models/cifar10_resnet18_lr_0.2_decay_0.0001_bsz_256_cosine_seed_20,SupContrast/save/SupCE/cifar10_models/cifar10_resnet18_lr_0.2_decay_0.0001_bsz_256_cosine_seed_30
tasks=("[0,1,2,3,4]" "[95,96,97,98,99]" "[6,11,16,21,26]" "[56,58,62,66,68]")

for task in "${tasks[@]}"
do
    python3 map_pretraining_logits.py -m task=cifar100 +map=supce map.ckpt_dir=$CKPT_SUPCE map.task_labs=$task deploy=True
done   






