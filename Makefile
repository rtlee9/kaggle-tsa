all: .make/preprocess .make/train .make/submission

.make/preprocess: src/preprocess.py src/input_pipeline.py
	python -m src.preprocess
	touch .make/preprocess

.make/train: src/train.py src/train_test_split.py src/utils.py src/model.py .make/preprocess
	python -m src.train -tz 1
	python -m src.train -tz 2
	python -m src.train -tz 3
	python -m src.train -tz 4
	python -m src.train -tz 5
	python -m src.train -tz 6
	python -m src.train -tz 7
	python -m src.train -tz 8
	python -m src.train -tz 9
	python -m src.train -tz 10
	python -m src.train -tz 11
	python -m src.train -tz 12
	python -m src.train -tz 13
	python -m src.train -tz 14
	python -m src.train -tz 15
	python -m src.train -tz 16
	touch .make/train

.make/submission: src/submission.py .make/train
	python -m src.submission
	touch .make/submission