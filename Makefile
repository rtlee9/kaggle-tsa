all: .make/preprocess .make/train .make/submission

.make/preprocess: src/preprocess.py src/input_pipeline.py
	python -m src.preprocess
	touch .make/preprocess

.make/train: src/train.py src/train_test_split.py src/utils.py src/model.py .make/preprocess
	python -m src.train
	touch .make/train

.make/submission: src/submission.py .make/train
	python -m src.submission
	touch .make/submission