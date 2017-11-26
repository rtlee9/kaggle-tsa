all: .make/preprocess .make/train .make/submission

.make/preprocess: src/preprocess.py
	python -m src.preprocess
	touch .make/preprocess

.make/train: src/train.py src/model.py src/pipeline.py src/zones.py src/constants.py src/config.py .make/preprocess
	python -m src.train -Z 16
	touch .make/train
