# Dataset Setup

The dataset files are excluded from Git. Download them locally and place them in the directories below.

## Classification data

Download the classification dataset from [Google Drive](https://drive.google.com/file/d/1WtjAJx4k_r2dI4YF4Ht84LcGDoerltQY/view), then extract its contents into:

```text
Data/classification/
```

## Detection data

Download the detection dataset from [Kaggle](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes?select=Train), then extract the `Train` and `Val` folders into:

```text
Data/detection/
```

The expected structure is:

```text
Data/detection/
├── Train/
│   ├── Glioma/
│   ├── Meningioma/
│   ├── No Tumor/
│   └── Pituitary/
└── Val/
    ├── Glioma/
    ├── Meningioma/
    ├── No Tumor/
    └── Pituitary/
```