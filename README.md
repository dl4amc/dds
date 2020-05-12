# Ensemble Wrapper Subsampling for Deep Modulation Classification

This repository contains source code used for the "Ensemble Wrapper Subsampling for Deep Modulation Classification" project

## Subsampling Techniques

- radioml_ews: Ensemble Wrapper Subsampling
- radioml_fisher: Subsampling using the Fisher Score
- radioml_fqi: Subsampling using the Feature Quality Index (FQI)
- radioml_laplacian: Sampling using the Laplacian Score
- radioml_rfs: Subsampling using Efficient and Robust Feature Selection (RFS)

## data
Download the RML2016.10b_dict.dat zip file (containing the original dataset) from www.kaggle.com/sramjee/rml10b and extract it in the data directory.

## models

Ranker Models used in the Subsampler Nets used in Ensemble Wrapper Subsampling.

## subsamplers

Subsampler Nets used in Ensemble Wrapper Subsampling. These subsampler nets use the Ranker Models in the models directory, perform the subsampling, and store data in the ranker_samples directory.

## ranker_samples

Ensemble Wrapper Subsampling samples selected using the CNN, CLDNN, and ResNet Subsampler Nets with 1/2, 1/4, 1/8, 1/16, and 1/32 subsampling rates.
Download the ranker_samples zip file (containing the subsampled data) from https://www.kaggle.com/sramjee/ranker-samples and extract it in the root (i.e. dds) directory.




