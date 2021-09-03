<p align="center">
  <a href="#"><img src="data/demo_dataset.gif" alt="emotic"></a>
</p>
<p align="center">
    <em>Emotion (Context + Facial) recognition in the wild using ConvNets (EfficientNet, ResNet, ResNext)</em>
</p>
<p align="center">
<a href="https://www.python.org/downloads/release/python-360/" style="text-decoration: none;" target="_blank">
    <img src="https://img.shields.io/badge/python-3.6-blue.svg" alt="License">
</a>

<a href="https://pypi.org/project/audioperm/" style="text-decoration: none;" target="_blank">
    <img src="https://img.shields.io/pypi/l/audioperm?style=flat" alt="License">
</a>

</p>

---
### Emotions In the Wild: CNN benchmarks
Emotion (Context + Facial) recognition in the wild using ConvNets (EfficientNet, ResNet, ResNext) 

For emotic dataset (pre-training),

* Download the dataset from http://sunai.uoc.edu/emotic/download.html

* Download the json annotations from https://www.kaggle.com/furcifer/emoticlabelsjson

* Install dependencies (`python >= 3.5`)
    ```console
    pip install -r requirements.txt
    ```
* If you want to use docker,
    ```
    nvidia-docker build -t nabil/efncv:eitw .
    nvidia-docker run -it -d -v /path_to_github_repo/EmotionsInTheWild-CNN-Benchmarks/:/eitw/ --net=host --ipc=host nabil/efncv:eitw /bin/bash
    ```

### Training

* Set the hyperparameters in the `config.py` file

* Run, `train.py`

### Pre-trained models

| Model           | Dataset       | pre-trained weight                                                                  |
|-----------------|---------------|-------------------------------------------------------------------------------------|
| resnet34        | emotic        | [resnet34](https://gitlab.com/zabir.al.nazi/emotions-in-the-wild-pretrained-models) |
| resnet50        | emotic + cfid | [resnet50](https://gitlab.com/zabir.al.nazi/emotions-in-the-wild-pretrained-models) |
| efficientnet-b0 | emotic + cfid | [efn-b0](https://gitlab.com/zabir.al.nazi/emotions-in-the-wild-pretrained-models)   |
| efficientnet-b1 | emotic + cfid | [efn-b1](https://gitlab.com/zabir.al.nazi/emotions-in-the-wild-pretrained-models)   |
| efficientnet-b2 | emotic + cfid | [efn-b2](https://gitlab.com/zabir.al.nazi/emotions-in-the-wild-pretrained-models)   |

### Support

> **Tested with:** `python3.6` `python3.7` `python3.8`

> **TO-DO:**
 - [x] augmentation
 - [ ] encoding facial landmarks

### Others
> Any contribution is welcome. 
  - [Contributors](https://github.com/zabir-nabil/EmotionsInTheWild-CNN-Benchmarks/graphs/contributors)


