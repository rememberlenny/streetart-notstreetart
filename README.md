![npaf-logo](https://user-images.githubusercontent.com/1332366/45318391-cb4e3200-b50a-11e8-8fcc-f64fce5c381b.png)

# Street art, not street art

Train a model that detects if image is or is not street art, based on images gathered from hashtagged content.

## What

The project above trains a model that detects whether an image is or is not street art. The model is trained on a image set gathered from hashtagged images for #streetart. The training data was compared against images from New York City. The image dataset was cleaned manually to have any mistagged content and NSFW images removed.

![Image montage](https://github.com/rememberlenny/streetart-notstreetart/blob/master/streetart_montage_v1.png?raw=true)

## Training results

Version one of the model and dataset, which was uncleaned, resulted in the following results:

![Training results](https://github.com/rememberlenny/streetart-notstreetart/blob/master/streetart_training_plot_v1.png?raw=true)

The latest training results can be seen on Comet.ml here: https://www.comet.ml/lenny/street-art-detection/dcec5a30912543839cc27ed30083cee2

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
│   ├── testing [generated from build_dataset.py]
│   │   ├── not_streetart [858 entries exceeds filelimit, not opening dir]
│   │   └── streetart [396 entries exceeds filelimit, not opening dir]
│   ├── training [generated from build_dataset.py]
│   │   ├── not_streetart [3124 entries exceeds filelimit, not opening dir]
│   │   └── streetart [1387 entries exceeds filelimit, not opening dir]
│   └── validation [generated from build_dataset.py]
│       ├── not_streetart [340 entries exceeds filelimit, not opening dir]
│       └── streetart [161 entries exceeds filelimit, not opening dir]
├── build_dataset.py
├── load_model.py
├── README.md
├── save_model.py
├── Street Art Detector.ipynb
├── streetart_model_v1.model
├── streetart_montage_v1.png
└── streetart_training_plot_v1.png
```

## How to run

1. `pip install -r requirements.txt`
2. Download dataset from Floydhub into `/dataset`. Folder structure for `/dataset/images` should match the format listed above.
3. Run `python build_dataset.py`. This will create the `/testing`, `/training`, and `/validation` dataset.
4. Run `python save_model.py` or use the python notebook and run the training step.
5. Use `python load_model.py` to validate the results.
