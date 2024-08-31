
# DATR: DOMAIN AGNOSTIC TEXT RECOGNIZER

#### DATR is a Student-Teacher-Assistant pipeline network with dual CLIP (ResNet50 and ViT image encoders) that improves cross-domain text recognition by fusing visual and textual features, outperforming existing models on diverse datasets.


### Installation

```
conda create -n DATR python=3.8
conda activate DATR
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

### Data
Download the Train/Val/Test split of all the datasets from [here.](https://drive.google.com/file/d/1bl4LZYc8IyH6uu1Nnmhrtc60vds8oLXx/view?usp=sharing)

- Extract the contents of data.zip into the 'data' directory located in the root of the project.


### Training

```
python train.py trainer.gpus=<num_gpus> ckpt_name=datr dataset=<dataset name (drone/underwater/natural)> model=datr model.batch_size=160 trainer.val_check_interval=1.0 trainer.max_epochs=5 model.lr=0.0014
```
### Model Results
 Domain (Trained on)  | Accuracy | Checkpoint |
| ------------- | ------------- | ------------- |
| Natural  | 93.34  | [Download]()  |
| Underwater  | 70.21  |[Download](https://drive.google.com/file/d/143arS2FhvIZx-bxXeqdJ35bg3I5Fj1x3/view?usp=sharing)  |
| Drone  | 98.92  |[Download](https://drive.google.com/file/d/1GbqFhTAhj1dVLpvAvyKHX0XbqtAJzjo7/view?usp=sharing)  |

### Testing

```
python test.py <path_to_checkpoint> --data_root data --test_set <test set name (natural/other)>
```


