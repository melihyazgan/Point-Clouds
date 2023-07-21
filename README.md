# Point-Clouds
This is the repository for the important implementations of processing/training of Lidar Data
## PointNet
PointNet is a popular deep learning architecture that can directly process unordered point clouds.   
It uses shared multi-layer perceptrons (MLPs) to learn local and global features from the points and combines them to perform object detection.   
You can find the pape here: https://arxiv.org/pdf/1612.00593.pdf. Here is the [Implementation](PointNet.ipynb).
## Voxelization
Question is Why do we need Voxelization?  
Let us first begin with Why do we need Voxelization.   
It is a preprocessing step in 3D point cloud processing and analysis Tasks.It involves dividing the 3D space into small volumetric elements called voxels, which can be thought of as 3D equivalents of pixels in 2D images.   
Voxelization is particularly useful for the following reasons:
- Voxelization provides a structured representation of unstructured 3D point clouds, enabling the use of traditional 3D convolutional neural networks.  
- It simplifies computation by aggregating points within voxels, reducing the data size for efficient processing.
- Voxelization allows for efficient 3D convolution, facilitating faster and scalable processing.
- Local feature extraction is enabled through voxel-wise aggregation of points, capturing important spatial information.
- Voxelized representations are memory-efficient, making it easier to handle large 3D point clouds.
- It helps handle sparse data, filling gaps in the point distribution for a more complete representation of the scene.

   
[Voxelnet Classifier](Classification_Voxelnet.ipynb) is developed with the functions they are developed in [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).Notebook can run without building OpenPCDet.
You can download the shapedataset from the [link](https://shapenet.org/)