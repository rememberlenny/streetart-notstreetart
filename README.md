![npaf-logo](https://user-images.githubusercontent.com/1332366/45318391-cb4e3200-b50a-11e8-8fcc-f64fce5c381b.png)

# Street art, not street art

Train a model that detects if image is or is not street art, based on images gathered from hashtagged content.

## What

The project above trains a model that detects whether an image is or is not street art. The model is trained on a image set gathered from hashtagged images for #streetart. The training data was compared against images from New York City. The image dataset was cleaned manually to have any mistagged content and NSFW images removed.

![Image montage](https://github.com/rememberlenny/streetart-notstreetart/blob/master/streetart_montage.png?raw=true)

## Training results

Version two of the model and dataset, resulted in the following results:

|	Name	| Value |	Min | Value |	Max | 
|-------|-------|-----|-------|-----|
|1 |	acc |	0.804197365509982 |	0.375 | 0.9375 |
|2 |	batch |	130	| 0 |	130 |
|3  |	loss | 0.5488157922014857 |	0.36985594034194946 	| 1.2533280849456787 |
|4 |	size | 32	| 31 |	32 |
|5  |	val_acc |	0.7505330491040562 |	0.7036247338567462 	| 0.7974413653680765 |
|6 |	val_loss |	0.6231901207204058 |	0.5713440334873159 	| 0.7559062163035075 |

![Training results](https://github.com/rememberlenny/streetart-notstreetart/blob/master/streetart_plot.png?raw=true)

The latest training results can be seen on Comet.ml here: https://www.comet.ml/lenny/street-art-detection/63f69003931a438abb477ae5c5bc4ca5

![screen shot 2019-01-19 at 11 59 36 pm](https://user-images.githubusercontent.com/1332366/51435496-502e2280-1c46-11e9-8f46-fa43763a1f33.png)

## Dataset

Dataset training dataset can be downloaded from Floydhub here:  https://www.floydhub.com/rememberlenny/datasets/streetart-notstreetart

To work correctly, save the dataset into the `/streetart` folder.

The correct directory structure should look like this:

```
├── pyimagesearch
│   ├── __pycache__
│   │   ├── config.cpython-36.pyc
│   │   ├── __init__.cpython-36.pyc
│   │   └── resnet.cpython-36.pyc
│   ├── config.py
│   ├── __init__.py
│   └── resnet.py
├── streetart
│   ├── images
│   │   ├── not_streetart [4322 entries exceeds filelimit, not opening dir]
│   │   └── streetart [1944 entries exceeds filelimit, not opening dir]
│   ├── testing
│   │   ├── not_streetart [858 entries exceeds filelimit, not opening dir]
│   │   └── streetart [396 entries exceeds filelimit, not opening dir]
│   ├── training
│   │   ├── not_streetart [3124 entries exceeds filelimit, not opening dir]
│   │   └── streetart [1387 entries exceeds filelimit, not opening dir]
│   └── validation
│       ├── not_streetart [340 entries exceeds filelimit, not opening dir]
│       └── streetart [161 entries exceeds filelimit, not opening dir]
├── build_dataset.py
├── not_streetart.png
├── README.md
├── requirements.txt
├── Street Art Detector.ipynb
├── streetart_model.model
├── streetart_montage.png
├── streetart_plot.png
├── test_model_by_generating_montage.py
└── train_model.py
```

## How to run

1. `pip install -r requirements.txt`
2. Download dataset from Floydhub into `/dataset`. Folder structure for `/dataset/images` should match the format listed above.
3. Run `python build_dataset.py`. This will create the `/testing`, `/training`, and `/validation` dataset.
4. Run `python train_model.py` or use the python notebook and run the training step.
5. Use `python test_model_by_generating_montage.py` to validate the results.
