Use the code in this repository to calculate the accuracy of the tree species fractions raster dataset. 
These scripts should be executed in the following order:

1. ratio_estimator.py --> derive the tree species fractions ground-truth based on sampling plots for reference polygons. Initial polygon based tree species fractions are mandatory! 
2. exactextract_raster_sampling.py --> sample the tree species fractions dataset and generate the prediction values. In Addition, coverage values can be sampled from a binary height raster.
3. gt_modeling.py --> derive the final ground truth for the polygons by combining the initial polygon species fractions and sample plot species fractions. Those are then normalised via the sampled coverage values.
4. gt_prediction_accuracy.py --> calculate the accuracy metrics from ground-truth and prediction values via fuzzy error matrices. 
