# Data
The dataset is not included in this repository.
Use the public Marathi handwritten dataset (2,520 images, MIT license)
and place the images in `data/images/`.
Create `data/labels.csv` with two columns:
image_path,text
data/images/img001.png,मराठी शब्द
Then run `python src/make_splits.py` to create the train/val/test files.
