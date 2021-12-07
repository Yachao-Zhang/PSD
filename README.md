# Perturbed Self-Distillation: Weakly Supervised Large-Scale Point Cloud Semantic Segmentation (ICCV2021)
This is the implementation of **PSD** (ICCV 2021), a simple weakly-supervised semantic segmentation of large-scale 3D point clouds. 
 
### (1) Setup
This code has been tested with Python 3.5, Tensorflow 1.13, CUDA 9.0 and cuDNN 7.4.1 on Ubuntu 16.04.

 - Clone the repository 
```
git clone --depth=1 https://github.com/Yachao-Zhang/PSD && cd PSD
```
- Setup python environment
```
pip install -r helper_requirements.txt
sh compile_op.sh
```

### (2) Weakly semantic Segmentation on S3DIS
S3DIS dataset can be found 
<a href="https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1">here</a>. 
Download the files named "Stanford3dDataset_v1.2_Aligned_Version.zip". Uncompress the folder and move it to 
`/data/S3DIS`.

- Preparing the dataset:
```
python utils/data_prepare_s3dis.py
```

Training and test of weakly semantic Segmentation on S3DIS Area-5 by:

```
sh jobs_s3dis_a5.sh 
```

### Citation
If you find our work useful in your research, please consider citing:

    @inproceedings{zhang2021perturbed,
        title={Perturbed Self-Distillation: Weakly Supervised Large-Scale Point Cloud Semantic Segmentation},
        author={Zhang, Yachao and Qu, Yanyun and Xie, Yuan and Li, Zonghao and Zheng, Shanshan and Li, Cuihua},
        booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
        pages={15520--15528},
        year={2021}
    }
    
A related work (Weakly Supervised Semantic Segmentation for Large-Scale Point Cloud AAAI-2021) can be found <a href="https://ojs.aaai.org/index.php/AAAI/article/view/16455">here</a>.
   ```
   @inproceedings{zhang2021weakly,
        title={Weakly Supervised Semantic Segmentation for Large-Scale Point Cloud},
        author={Zhang, Yachao and Li, Zonghao and Xie, Yuan and Qu, Yanyun and Li, Cuihua and Mei, Tao},
        booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
        volume={35},
        number={4},
        pages={3421--3429},
        year={2021}
    }
    ```
    
### Acknowledgment
Note that this code is heavily borrowed from RandLA-Net (https://github.com/QingyongHu/RandLA-Net).
