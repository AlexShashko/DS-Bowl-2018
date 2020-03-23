# DS-Bowl-2018
Prediction for 2018 Data Science Bowl challange.
# Overview
Semantic segmentation UNet using Keras. All needed data and more information can be found at https://www.kaggle.com/c/data-science-bowl-2018/data
## Folder structure
Download and unzip repository files and data files in one directory to achive next structure:
<pre>
ds-bowl-2018/
	stage1_test/
	stage1_train/
	main_task.ipynd
	model-weights.h5
	predick_mask.py
	requirements.txt
	train.py
</pre>
## Preprocessing
In preprocessing phase images are resized to 128x128 resolution and masks are collected in one image same resolution.
## Model
This model is basic structure of UNet model (details about UNet - https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
