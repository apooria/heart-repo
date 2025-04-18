# Heart Training Codes

## Installing the dependencies
To install the dependecies, you can make a conda environment and pip-install the packages.
```bash
conda create -n heart-env python=3.12
conda activate heart-env
pip install -r requirements.txt
```

## Downloading required data
You can access the data with the following commands:
```bash
mkdir data
aws s3 sync s3://pooria-personal/organ_age/data_factory/ ./data/
```
This data includes the following folers:
+ `metadata`: including patient demographics for ~100k patints, normal heart flag based on 42k pickle report, and a dummy regression file which I used from my previous runs to train the volume filtering regression model.
+ `regression models`: includes the linear regression model used for volume filtering which is train to predict heart volume vs [age, bmi, weight, sex], and the stats including mean and std from the dummy regression dataset. The model should work pretty descent, but can also be trained based on new data (check the part for fit and save in `notebooks/processed_data_filtering.ipynb`).
+ `dummy_data`: 100 images to test the scripts.
+ `samples`: sampled data for training and testing (check `data_sampling.py/ipynb` for details).

## Preprocessing
The `preprocess.py` scripts, masks each image around the centroid of the heart with specific distances in the physical space. It reads the data from `data/dummy_data/` to the variable `data_source`. The convention I used is to have nifti images in `images` and nifti masks in `masks` folders in the `data_source` original folder.

Based on the `dataset_id` (which you can specify yourself, I just wrote `processed_dummy_1`) in the begining of the main function, it makes a folder inside the `data` dirctory and saves the mask-resampled images, masks, and some additional information. 

To run the script, just do:
```bash
cd scripts
python preprocess.py
```
The processed data directory would look like this:
```bash
+ data
    + dataset_id
        + images
            + img1.nii.gz
            + ...
        + masks
            + msk1.nii.gz
            + ...
        + info
            + info.csv
```

The additional info (`info.csv`) has the following information:
+ Study ID
+ image and mask directory (inside `images` and `masks` folders)
+ Demographics (based on the 100k patient information file)
+ Heart volume

## Filter larger/smaller volumes
I have already trained the regression model and saved inside `data/regression_models/lm_volume.pkl`. The `filter.py` script reads the weights and based on the volume numbers in `info.csv`, filters the heart volumes. You can run filtering with the following command:
```bash
cd scripts
python filter.py
```
After running this, the data folder looks like this:
```bash
+ data
    + dataset_id
        + images
            + img1.nii.gz
            + ...
        + masks
            + msk1.nii.gz
            + ...
        + info
            + info.csv
            + normal_heart.csv
            + large_heart.csv
            + small_heart.csv
```

## Training code
You can run:
```bash
cd scripts
python train.py
```
You can change the parameters in the defined config variable in the begining of the main function. There are two important variables:
+ `data_id`: the same variable we used to proceprocess data (like `processed_dummy_1`). Input data is coming from `data/data_id/`.
+ `result_id`: the name of the folder which the results will be saved.

The training script, reads the data from `data/data_id/info/normal_heart.csv` and uses it to generate torch datasets (`HeartDataset` class).
After running the training, the results directory will look like this:
```bash
+ results
    + result_id (like run_dummy_1)
        + checkpoints/
        + config.json
        + loss_results.csv
        + train_data.csv
        + valid_data.csv
```


## Bonus Tip: Dataset sampling
Sampling generates a list of study ids with related metadata from 100k patient information file inside `data/metadata` directory. The dataset will be balance on sex-age group classes (like male:20-30, female:20-30, ...)
To run the sampling, do:
```bash
cd scripts
python data_sampling.py
```
You can change the process using the following parameters in the begining of the main function:
```python
SEED = 42
TRAIN_SAMPLE_SIZE = 700  # size for each sex-age group [male:20-30, male-30-40, ...]
TEST_SAMPLE_SIZE = 200  # size for each sex-age group
```
This scrips saves two files in the `data/samples/`:
+ `train_data.csv`
+ `test_data.csv`

## TODO
+ Large-scale inference script