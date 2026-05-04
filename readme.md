# Memory-Augmented Query Intent Understanding for Efficient Chat-based Image Retrieval

## Datasets
Please download all datasets from their respective official websites.
- COCO 2017 Unlabeled Images
- VisDial

```
- VisDial
  ├── train
  │   ├── images
  │   └── visdial_1.0_train.json
  └── val
      ├── images
      └── visdial_1.0_val.json
```
After preparing the datasets above, please run the following script to extract the image features for efficient training:

```
python prepare_datas.py
```
## Environments

- **Ubuntu** 20.04  
- **CUDA** 12.6  
- **Python** 3.10  

Use the following instructions to create the corresponding conda environment.  
Please make sure to download the required pretrained models (e.g., BLIP) from their official sources before running the training or evaluation scripts.

```
conda create --name maqiu python=3.10 -y
conda activate maqiu
pip install -r requirements.txt
```
## Training and Evaluation

### Run the following script for multi-GPU finetuning on VisDial:

```
./train.sh $run_id
```

**Argument meaning**

- `$run_id` — folder name for saving checkpoints and logs


**Example**

```
./train.sh run_0
```

### Run the following script for evaluation:

```
./eval.sh $run_id
```

**Example**

```
./eval.sh run_0
```



