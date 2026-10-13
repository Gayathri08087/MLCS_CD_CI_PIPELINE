.PHONY: install format train eval all

install:
	python -m pip install -r requirements.txt

format:
	black --check train.py

train:
	python train.py

eval:
	cat results/metrics.txt

all: install format train eval
