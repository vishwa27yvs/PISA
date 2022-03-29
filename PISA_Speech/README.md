
### EISA and PISA


1. Install the required libraries from
```
!pip install -r requirements.txt
```
This also installs the geoopt library (https://github.com/geoopt/geoopt) to run the necessary hyperbolic functions
2. Dataset generation

dataset_gen folder provides script for processing the datasets after download

3. Example commands

### EISA
```
!python main.py --dataset <dataset_name> --netType envnetv2 --data <path_to_dataset_folder> --strongAugment --nEpoch <num_epochs> --LR 0.01 --hyp_mean 0 --save <save_model_path>
```

### PISA - uses hyperbolic midpoint

```
!python main.py --dataset <dataset_name> --netType envnetv2 --data <path_to_dataset_folder> --strongAugment --nEpoch <num_epochs> --LR 0.01 --hyp_mean 1 --save <save_model_path>
```
