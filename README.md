## Environment Setup
Run the following commands to create the Conda environment:
```
conda env create -f environment.yml
conda activate cxr_bias
```

## Training a Single Model
To train a single model, use `train_model.py` with the appropriate arguments, for example:
```
python train_model.py \
     --domain MIMIC \
     --use_pretrained \
     --target race \
     --data_type ifft \
     --model densenet \
     --filter_type high \
     --filter_thres 100 
```


## Training a Grid of Models
To reproduce the experiments from the paper, use `sweep.py` with the experiment grids defined in `experiments.py`, for example:
```
python sweep.py \
     --output_dir=/my/sweep/output/path\
     --command_launcher slurm\
     --slurm_pre slurm_arguments
     --experiment IFFTPatched 
```


We provide the bash script used for our main experiments in the scripts directory. You will need to customize them, along with the launcher, to your compute environment.

## Aggregating Results

We provide sample code for creating aggregate results after running all experiments in `AggResults.ipynb`.

