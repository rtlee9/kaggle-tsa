all: .make/preprocess .make/train .make/submission

.make/preprocess: src/preprocess.py
	python -m src.preprocess
	touch .make/preprocess

.make/train: src/train.py .make/preprocess
	python -m src.train
	touch .make/train
