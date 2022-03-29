

### PISA


- The datsets CIFAR 10 and STL10 will be automatically downloaded during training from load_data.py from torchvision datasets
- The hyperbolic functions are present in main.py


To execute the code:

We need to install the geoopt library (https://github.com/geoopt/geoopt) to run the necessary hyperbolic functions
```
!pip install -r requirements.txt
```
Example command:
```
!python main.py --dataset stl10 --data_dir <path_to_dataset> --root_dir <path_to_save_output> --labels_per_class <n> --arch preactresnet18  --learning_rate 0.2 --momentum 0.9 --decay 0.0001 --epochs 600 --schedule 350 500 --gammas 0.1 0.1 --train mixup --mixup_alpha 1.0 --graph True --n_labels 3 --eta 0.2 --beta 1.2 --gamma 0.5 --neigh_size 4 --transport True --t_size 4 --t_eps 0.8 --seed 42
```
