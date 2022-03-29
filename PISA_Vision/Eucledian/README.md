
====
Eucledian methods
====

The datsets CIFAR 10 and STL10 will be automatically downloaded during training from load_data.py from torchvision datasets

Example commands on how to execute each model


## PuzzleMix

!python main.py --dataset <name_stl10,cifar10>  --data_dir <path_to_dataset> --root_dir <path_to_save_output> --labels_per_class <n> --arch preactresnet18  --learning_rate 0.2 --momentum 0.9 --decay 0.0001 --epochs 600 --schedule 350 500 --gammas 0.1 0.1 --train mixup --mixup_alpha 1.0 --graph True --n_labels 3 --eta 0.2 --beta 1.2 --gamma 0.5 --neigh_size 4 --transport True --t_size 4 --t_eps 0.8


## Manifold mixup
!python main.py --dataset <name_stl10,cifar10> --data_dir <path_to_dataset> --root_dir <path_to_save_output> --labels_per_class <n> --arch preactresnet18  --learning_rate 0.2 --momentum 0.9 --decay 0.0001 --epochs 600 --schedule 350 500 --gammas 0.1 0.1 --train mixup_hidden --mixup_alpha 1.0

## Input mixup

!python main.py --dataset <name_stl10,cifar10>  --data_dir <path_to_dataset> --root_dir <path_to_save_output> --labels_per_class <n> --arch preactresnet18  --learning_rate 0.2 --momentum 0.9 --decay 0.0001 --epochs 600 --schedule 350 500 --gammas 0.1 0.1 --train mixup --mixup_alpha 1.0


## PreActResNet18

!python main.py --dataset <name_stl10,cifar10>  --data_dir <path_to_dataset> --root_dir <path_to_save_output> --labels_per_class <n> --arch preactresnet18  --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 1200 --schedule 400 800 --gammas 0.1 0.1 --train mixup --mixup_alpha 1.0 --graph True --n_labels 3 --eta 0.2 --beta 1.2 --gamma 0.5 --neigh_size 4 --transport True --t_size 4 --t_eps 0.8
