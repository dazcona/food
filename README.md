# Transfer Learning with Keras and Deep Learning in Python

There are two types of transfer learning in the context of deep learning:

1. Transfer learning via feature extraction
2. Transfer learning via fine-tuning

In this example, we will be treating networks as arbitrary feature extractors. When performing feature extraction, we treat the pre-trained network as an arbitrary feature extractor, allowing the input image to propagate forward, stopping at pre-specified layer, and taking the outputs of that layer as our features.

## Deployment

1. Download food data and put it in data/:
```
$ wget --passive-ftp --prefer-family=ipv4 --ftp-user FoodImage@grebvm2.epfl.ch \
	--ftp-password Cahc1moo ftp://tremplin.epfl.ch/Food-5K.zip
$ unzip Food-5k.zip
```

2. Build the docker image:
$ cd docker
$ make build

3. Create a docker container based on the image:
$ make run

4. SSH to the docker container:
$ make dev

5. Build our custom dataset:
```
$ python src/build_dataset.py
```

6. Extract features:
```
$ python src/extract_features.py
```

7. Train:
```
$ python src/train.py
```

## Resources

* https://www.pyimagesearch.com/2019/05/20/transfer-learning-with-keras-and-deep-learning/