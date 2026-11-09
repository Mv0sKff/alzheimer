# Alzheimer's classification

## Install

```sh
pip install -r requirements.txt
```

## Download dataset

[OASIS Alzheimer's Detection Dataset](https://www.kaggle.com/datasets/ninadaithal/imagesoasis)

Expected layout: `data/<class>/*.jpg` (for example `data/Non Demented/`).

## Generate CSV

```sh
python make_csv.py
```

Writes subject-level splits to `data/csv/`: `test.csv`, `train_fold_<k>.csv`, `val_fold_<k>.csv`.

## Train

```sh
python main.py --backbone resnet18 --epochs 50
```

Results go to `sessions/<timestamp>/`. Resume with `--resume sessions/<timestamp>`.

## Evaluate

```sh
python evaluate.py sessions/<timestamp>
```

Writes `test_results.csv` and `test_predictions.csv` to the run directory.
