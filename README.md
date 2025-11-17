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

# Datasets
We provide subset in images_un.

All data can be installed from [ACA](https://github.com/Omenzychen/Adversarial_Content_Attack)

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


