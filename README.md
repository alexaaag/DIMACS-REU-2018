
In this folder, you’ll find 

- quality_metrics.Rmd: R code that defines profile residual.
- modeling.Rmd: R code of the classifiers. It includes K-means clustering, PCA and Random Forest.
- measurements folder: all the measurement matrices
- umpca folder: ump related matlab code and reconstruction
- qualitydf.csv: the quality df exported from quality_metrics.Rmd

Within umpca:
- fea_extraction_umpca.m: matlab code that calculates features extracted from high VR data
- matrices folder: all VR measurement matrices. They are exported from modeling.Rmd
- mexeig.m, UMPCA.m, tensor_toolbox: UMPCA toolbox
- reconstruction: reconstructed matrices exported from matlab code and R code that visualizes the reconstruction
- newfea_test.csv, newfea_train.csv: features extracted
