all: .make/preprocess .make/train .make/submission

.make/preprocess: src/preprocess.py
	python -m src.preprocess
	touch .make/preprocess

.make/train: src/train.py src/model.py src/pipeline.py src/zones.py src/constants.py src/config.py .make/preprocess
	python -m src.train -Z 16
	python -m src.train -Z 14
	python -m src.train -Z 12
	python -m src.train -Z 10
	python -m src.train -Z 7
	python -m src.train -Z 3
	python -m src.train -Z 4
	python -m src.train -Z 9
	python -m src.train -Z 5
	python -m src.train -Z 17
	touch .make/train
