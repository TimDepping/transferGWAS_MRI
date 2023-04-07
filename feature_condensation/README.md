# Create embeddings

This directory contains code to condense images into low-dimensional embeddings.

The scripts `clean_individuals.py` and `feature_condensation.py` are getting called by the `pipeline/feature_condensation.slurm` script that is part of the pipeline. 
The other scripts are run in addition to the pipeline.

## Export image embeddings at specified layer in convolutional neural network

Run via
```bash
python feature_condensation/feature_condensation.py PATH_TO_CSV OUT_DIR \
                --n_pcs 50 \
                --model resnet50 \
                --pretraining models/50_channel_model.pt
                --layer L4
                --use_mc
```
where `PATH_TO_CSV` is a `.csv` files with `IID` and `path` column (see `Input format` below), and `OUT_DIR` is the directory to save the results.


### Input format

If you have a single image per individual, simply create a `.csv` file with an `IID` and `path` column as in this example:
```CSV
IID,path
1234567,path/to/img1.png
2345678,path/to/img2.png
3456789,path/to/img3.png
4567890,path/to/img4.png
...
```

If each individual has more than one image (such as one image per eye, multiple slices from a scan, ...), create an additional `instance` column. Note that all individuals need to have the same number of instances (otherwise they will be sorted out).
```CSV
IID,path,instance
1234567,path/to/left_img1.png,left
1234567,path/to/right_img2.png,right
2345678,path/to/left_img3.png,left
2345678,path/to/right_img4.png,right
```

Images must be in a standard image format that `PIL` can read (such as `.png` or `.jpg`). In addition, we adapted the [original script by Matthias Kirchler](https://github.com/mkirchler/transferGWAS/blob/master/feature_condensation/feature_condensation.py) so that is can use multi channel images as `.pt` files by using the `--use_mc` paramenter. For our case, this files contain all 50 images for one patient.