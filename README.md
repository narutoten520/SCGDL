# SCGDL
Spatial Clustering for Spatial Omics Data via Geometric Deep Learning

The proposed SCGDL owns a two-levels architecture, as shown in Figure 1. 
The feature and adjacent matrices are regarded as two incentives in the higher-level framework.
Based on the built-in gene expression profiles, the feature matrix is calculated. It indi-cates the inclusion 
relation of spots and genes. The adjacent matrix is derived according to the positional information of spots.
A spa-tial neighbor graph (SNG) is capable of being delineated during the generation of an adjacent matrix. 
These two inputs are conducted by a DGI module with four layers of RGGCNN. 
Finally, the low-dimensional latent embeddings are acquired to imply the spots representation at the higher-level.
