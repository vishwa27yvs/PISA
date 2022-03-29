
Dataset setup
===============

- Make a directory to save datasets.
```
mkdir datasets/[dataset_name]/unprocessed
```
- Store all .wav files or folders containing .wav files (depending on the dataset) in the unprocessed folder created
- Datasets can be downloaded from the sources mentioned in the paper. Language dataset mapping (English:SAVEE,Italian:EMOVO,German:EmoDB,Persian:ShEMO, Urdu: Urdu SER )

- Run any of the files as required to generate wav16.npz and wav44.npz for the dataset, e.g.,
```
python dataset_gen/[file_name]
```
