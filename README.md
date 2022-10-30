# Inverse Rendering for Complex Indoor Scenes: <br> Shape, Spatially-Varying Lighting and SVBRDF <br> From a Single Image <br> ([Project page](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/))
Zhengqin Li, Mohammad Shafiei, Ravi Ramamoorthi, Kalyan Sunkavalli, Manmohan Chandraker

This repo is a developer version of the original repo by Zhengqin Li, for his paper (***Li'20**) Li, Zhengqin, et al. "Inverse rendering for complex indoor scenes: Shape, spatially-varying lighting and svbrdf from a single image." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.

This repo has the following features beyond the original repo:
- cleaned-up scripts with a **modularized** design (e.g. training, logging, checkpoing, and of sourse, models);
- a new YAML-based config system;
- tools for debugging, logging and visualization;
- tools for running on the UCSD PRP cluster.
- tested for OpenRooms;
- included other inverse rendering models (e.g. IRISFormer, Zhu et al., 2022 (***Zhu'22**) `[TODO]`)

Not ready:

- [] cascade refinement, with cascade > 1
- [] test on real images (real images, IIW, nyud)
<!-- - lighting data on the cluster and data loading of it -->

## Project folder structure
- **train**
  - **configs** # config files to combine (overload) the default ones
  - **utils**
    - config
      - defaults.py # the default config file
  - **data** # dataset-specific files
  - **models_def** # model class definitions
- **checkpoints** # for PyTorch ckpts
- **logs** # for logs including TensorBoard
- **summary_vis** # for dumped visualizations, pickle, etc.
- **models_ckpt** # for pre-trained checkpoints (see below for download links)
- **clean_up_*.py** # for use to clean up files from history tasks

Notes:

train/models.py # file from the original model (https://github.com/lzqsd/InverseRenderingOfIndoorScene/blob/master/models.py); keep it here so that *torch.load* can function normally to load the original checkpoints

## Dataset structure
### OpenRooms public release
`DATASET.dataset_path_local`:
- main_xml | main_xml1 | mainDiffLight_xml | mainDiffLight_xml1 | mainDiffMat_xml | mainDiffMat_xml1 # 6 splits
  - scene0552_00 # thousands of scenes
    - im_%d.png/im_%d.hdr # RGB (PNG is only for reference; scales may be inconsistent)
    - imdepth_%d.dat # depth maps
    - imnormal_%d.png # normals in 2D
    - imroughness_%d.png # roughness in 2D
    - imbaseColor_%d.png # albedo in 2D
    - imenv_%d.hdr # per-pixel lighting envmaps in 2D

## To run locally (if you have GPU-enabled Linux machine)

Have a copy of dataset locally (if you are not on Hyperion/Theia):
- Mini dataset for local debugging (overfitting to)
  - Download [pngs images](https://drive.google.com/drive/folders/1-8RChRrXRO4F1HJv-UgaCucimihc9amy?usp=sharing) and change path in defaults.py (i.e. `_C.DATASET.png_path_mini_local = '/data/ruizhu/ORmini-pngs'`)
  - Download [other data](https://drive.google.com/drive/folders/1otm31GBHdmTTsyjbzGRqOLU4eyBwJ63s?usp=sharing) and change path in defaults.py (i.e. `_C.DATASET.dataset_path_mini_local = '/data/ruizhu/openrooms_mini'`)
  - [NOTE] To use mini for training, you need to set `DATASET.mini True` in your script
- full dataset
  - Download [RAW dataset for OpenRooms](https://ucsd-openrooms.github.io/) and organize under `DATASET.dataset_path_local` (see previous section)

Set paths in train/utils/config/defaults.py:

- `PATH.root_local` to the {project path} to which you cloned the project into.

Download pretrained models to `models_ckpt`:

- [models](https://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/models.zip) from *Li et al., 2020, Inverse Rendering for Complex Indoor Scenes*
- [models]() from *Zhu et al., 2022, IRISFormer* [TODO]

Install Conda and create environment:

<!-- https://docs.conda.io/en/latest/miniconda.html -->
<!-- wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -->
<!-- bash Miniconda3-latest-Linux-x86_64.sh # choose Miniconda installation path as /{}data/miniconda3 -->

```bash
conda create -n python=3.8
conda activate py38

conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
git clone https://github.com/Jerrypiglet/indoorInverse.git
pip install -r requirements.txt
```

Demo test script which loads pretrained model by *Li'20*.
```bash
(py38) ruizhu@ubuntu:~/Documents/Projects/indoorInverse$ conda activate py38 && CUDA_VISIBLE_DEVICES=0 python train/trainBRDFLight.py --task_name test_BRDF_locally --if_train False --if_val True --if_vis True --config openrooms_test_BRDF.yaml DATASET.mini True
```
### To run on the UCSD-prp cluster
See [here](extra_files/README_ucsd-prp.md) for instructions on running the code on the UCSD Pacific Research Platform(PRP) cluster.
## Notes by Rui
  
Training script used by Rui:
```bash
python rui_tool.py create -d --gpus 1 -f rui_torch_job_2gpu_v6_mclab.yaml --memr 20 --meml 50 --cpur 15 --cpul 30 -s 'python -m torch.distributed.launch --master_port 5320 --nproc_per_node=1 trainBRDFLight.py --if_cluster --task_name DATE-train_POD_trainBRDF_scratch --if_train True --if_val True --if_vis True --eval_every_iter 5000 --if_overfit_train False DATASET.num_workers 16 MODEL_BRDF.enable True MODEL_BRDF.load_pretrained_pth False MODEL_BRDF.enable_BRDF_decoders True MODEL_BRDF.enable_list al_de_no_ro MODEL_BRDF.loss_list al_de_no_ro DATA.data_read_list al_de_no_ro DATA.im_height 240 DATA.im_width 320 SOLVER.ims_per_batch 8 TEST.ims_per_batch 8 DATA.if_load_png_not_hdr True DATASET.mini False MODEL_BRDF.load_pretrained_pth False'
```

# From original README file
## 2. Useful links:
* Project page: http://cseweb.ucsd.edu/~visco…/projects/CVPR20InverseIndoor/
* Trained models: http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/models.zip
* Renderer: https://github.com/lzqsd/OptixRenderer
* Object insertion: https://github.com/lzqsd/VirtualObjectInsertion
* Tileable texture synthesis: https://github.com/lzqsd/TileableTextureSynthesis
* Spherical gaussian optimization: https://github.com/lzqsd/SphericalGaussianOptimization
* Dataset: https://ucsd-openrooms.github.io/ 

## 3. Results on our new dataset 
This is the official code release of paper [Inverse Rendering for Complex Indoor Scenes: Shape, Spatially-Varying Lighting and SVBRDF From a Single Image](https://drive.google.com/file/d/18zG1kzVpL9XsEVBK95hbpnB-FMlChRXP/view). The original models were trained by extending the SUNCG dataset with an SVBRDF-mapping. Since SUNCG is not available now due to copyright issues, we are not able to release the original models. Instead, we rebuilt a new high-quality synthetic indoor scene dataset and trained our models on it. We will release the new dataset in the near future. The geometry configurations of the new dataset are based on  ScanNet [1], which is a large-scale repository of 3D scans of real indoor scenes. Some example images can be found below. A video is at this [link](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/github/dataset.mp4)
![](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/github/dataset.png)
Insverse rendering results of the models trained on the new datasets are shown below. 
![](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/github/inverseRendering.png)
Scene editing applications results on real images are shown below, including results on object insertion and material editing.
![](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/github/objectInsertion.png)
![](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/github/materialEditing.png)
Models trained on the new dataset achieve comparable performances compared with our previous models. Quantitaive comparisons are listed below, where [Li20] represents our previous models trained on the extended SUNCG dataset. 
![](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/github/quantitative.png)

## 4. Download the trained models
The trained models can be downloaded from the [link](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/models.zip). To test the models, please copy the models to the same directory as the code and run the commands as shown below. 

## 5. Train and test on the synthetic dataset
To train the full models on the synthetic dataset, please run the commands
* `python trainBRDF.py --cuda --cascadeLevel 0 --dataRoot DATA`: Train the first cascade of MGNet. 
* `python trainLight.py --cuda --cascadeLevel 0 --dataRoot DATA`: Train the first cascade of LightNet.
* `python trainBRDFBilateral.py --cuda --cascadeLevel 0 --dataRoot DATA`: Train the bilateral solvers.
* `python outputBRDFLight.py --cuda --dataRoot DATA`: Output the intermediate predictions, which will be used to train the second cascade. 
* `python trainBRDF.py --cuda --cascadeLevel 1 --dataRoot DATA`: Train the first cascade of MGNet. 
* `python trainLight.py --cuda --cascadeLevel 1 --dataRoot DATA`: Train the first cascade of LightNet.
* `python trainBRDFBilateral.py --cuda --cascadeLevel 1 --dataRoot DATA`: Train the bilateral solvers.

To test the full models on the synthetic dataset, please run the commands
* `python testBRDFBilateral.py --cuda --dataRoot DATA`: Test the BRDF and geometry predictions.
* `python testLight.py --cuda --cascadeLevel 0 --dataRoot DATA`: Test the light predictions of the first cascade.
* `python testLight.py --cuda --cascadeLevel 1 --dataRoot DATA`: Test the light predictions of the first cascade.

## 6. Train and test on IIW dataset for intrinsic decomposition
To train on the IIW dataset, please first train on the synthetic dataset and then run the commands:
* `python trainFineTuneIIW.py --cuda --dataRoot DATA --IIWRoot IIW`: Fine-tune the network on the IIW dataset.

To test the network on the IIW dataset, please run the commands
* `bash runIIW.sh`: Output the predictions for the IIW dataset.
* `python CompareWHDR.py`: Compute the WHDR on the predictions.

Please fixing the data route in `runIIW.sh` and `CompareWHDR.py`. 

## 7. Train and test on NYU dataset for geometry prediction
To train on the BYU dataset, please first train on the synthetic dataset and then run the commands:
* `python trainFineTuneNYU.py --cuda --dataRoot DATA --NYURoot NYU`: Fine-tune the network on the NYU dataset.
* `python trainFineTuneNYU_casacde1.py --cuda --dataRoot DATA --NYURoot NYU`: Fine-tune the network on the NYU dataset.

To test the network on the NYU dataset, please run the commands
* `bash runNYU.sh`: Output the predictions for the NYU dataset.
* `python CompareNormal.py`: Compute the normal error on the predictions.
* `python CompareDepth.py`: Compute the depth error on the predictions.

Please remember fixing the data route in `runNYU.sh`, `CompareNormal.py` and `CompareDepth.py`.

## 8. Train and test on Garon19 [2] dataset for object insertion
There is no fine-tuning for the Garon19 dataset. To test the network, download the images from this [link](https://lvsn.github.io/fastindoorlight/). And then run `bash runReal20.sh`. Please remember fixing the data route in `runReal20.sh`.

All object insertion results and comparisons with prior works can be found from this [link](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/objectInsertion.zip). The code to run object insertion can be found from this [link](https://github.com/lzqsd/VirtualObjectInsertion). 

## 9. Differences from the original paper
The current implementation has 3 major differences from the original CVPR20 implementation.
* In the new models, we do not use spherical Gaussian parameters generated from optimization for supervision. That is mainly because the optimization proceess is time consuming and we have not finished that process yet. We will update the code once it is done. The performance with spherical Gaussian supervision is expected to be better.
* The resolution of the second cascade is changed from 480x640 to 240x320. We find that the networks can generate smoother results with smaller resolution.
* We remove the light source segmentation mask as an input. It does not have a major impact on the final results.

## 10. Reference 
[1] Dai, A., Chang, A. X., Savva, M., Halber, M., Funkhouser, T., & Nießner, M. (2017). Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 5828-5839).

[2] Garon, M., Sunkavalli, K., Hadap, S., Carr, N., & Lalonde, J. F. (2019). Fast spatially-varying indoor lighting estimation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 6908-6917).
