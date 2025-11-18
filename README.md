# APA


## Requirements
* Python3
* pytorch >=2.1.0 and torchvision
* diffusers
* peft
* transformers
* accelerate
  


# Model
```
cd model_ckpt
bash install.sh
```
Installed from [ACA](https://github.com/Omenzychen/Adversarial_Content_Attack)

# Datasets
We provide subset in images_un.

All data can be installed from [there](https://github.com/VL-Group/Natural-Color-Fool/releases/download/data/images.zip)

## Attack
Visual Consistency Alignment
```
python visual_alignment.py
```

Attack Effectiveness Alignment

```
#skip-gradient
python attack_alignment.py --gradient_back skip-gradient

#gradient checkpoint
python attack_alignment.py --gradient_back gc
```


