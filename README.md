
# SCGDL
![](https://github.com/narutoten520/SCGDL/blob/b839861ed8eb84e0d7de892d82fb07b7818301cd/Figure%201-updated.png)
Spatial Clustering for Spatial Omics Data via Geometric Deep Learning
------
The proposed SCGDL owns a two-levels architecture, as shown in Figure 1. 
The feature and adjacent matrices are regarded as two incentives in the higher-level framework.
Based on the built-in gene expression profiles, the feature matrix is calculated. It indi-cates the inclusion 
relation of spots and genes. The adjacent matrix is derived according to the positional information of spots.
A spa-tial neighbor graph (SNG) is capable of being delineated during the generation of an adjacent matrix. 
These two inputs are conducted by a DGI module with four layers of RGGCNN. 
Finally, the low-dimensional latent embeddings are acquired to imply the spots representation at the higher-level.
## Contents
* [Prerequisites](https://github.com/narutoten520/GRAPHDeep/edit/main/README.md#prerequisites)
* [Example usage](https://github.com/narutoten520/GRAPHDeep/edit/main/README.md#example-usage)
* [Trouble shooting](https://github.com/narutoten520/GRAPHDeep/edit/main/README.md#trouble-shooting)

### Prerequisites

1. Python (>=3.8)
2. Scanpy
3. Squidpy
4. Pytorch_pyG
5. matplotlib
6. torch_geometric

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Example usage
* Using SCGDL to identify the spatial domains for spatial transcriptome
  ```sh
    running SCGDL_Tutorial.ipynb to see the simulation results step by step
  ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Trouble shooting

* data files<br>
Please down load the spatial transcriptomics data from the provided links.

* Porch_pyg<br>
Please follow the instruction to install pyG and geometric packages.
