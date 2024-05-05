# HMPR: A Hierarchical and Multi-modal Framework for Place Recognition with a Learnable Metric <br />


## Description
This is the source code for paper

> C. Shu and Y. Luo. A Hierarchical and Multi-modal Framework for Place Recognition with a Learnable Metric,
> IEEE Transactions on Intelligent Vehicles.


## Usage
1. build the ops
  ```bash
  cd libs/pointops && python setup.py install && cd ../../
  ```
2. prepare the data <br/>
We use the processed point cloud submaps from [PointNetVLAD](https://arxiv.org/abs/1804.03492). <br/>
The corresponding RGB images are retrieved from center camera of the original Oxford RobotCar dataset according to the closest timestamps.<br/>
Then RGB images are downsampled to 320*240 resolution.

3. generate training queries
```bash
cd generate_queries && python generate_training_tuples.py
```
4. training in first step
```bash
cd train && python train_coarse.py --work_path ../log/coarse_net --config_path ../config/config_coarse.yaml
```
4. generate top K candidate
```bash
cd generate_queries && python generate_topK_candidates.py
```
5. training in refinement step
```bash
cd train && python train_fine.py --work_path ../log/fine_net --config_path ../config/config_fine.yaml
```

## Acknowledgements
Code is built based on [AdaFusion](https://github.com/MetaSLAM/AdaFusion), [PointNetVLAD](https://github.com/mikacuy/pointnetvlad), [PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch), [PointWeb](https://github.com/hszhao/PointWeb) and [CVTNet](https://github.com/BIT-MJY/CVTNet).



## Citation
If you find our work useful in your research, please consider citing:

    @ARTICLE{10508992,
    author={Shu, Chengfu and Luo, Yutao},
    journal={IEEE Transactions on Intelligent Vehicles},
    title={A Hierarchical and Multi-modal Framework for Place Recognition with a Learnable Metric},
    year={2024},
    pages={1-10},
    doi={10.1109/TIV.2024.3394213}}
