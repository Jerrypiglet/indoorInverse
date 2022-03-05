m# Inverse Rendering for Complex Indoor Scenes: <br> Shape, Spatially-Varying Lighting and SVBRDF <br> From a Single Image <br> ([Project page](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/))
Zhengqin Li, Mohammad Shafiei, Ravi Ramamoorthi, Kalyan Sunkavalli, Manmohan Chandraker
## 1.1 To run locally (if you have GPU-enabled Linux machine)
Checklist:
- Have a copy of dataset locally (if you are not on Hyperion/Theia):
  - mini dataset for local debugging (overfitting to)
    - Download [pngs images](https://drive.google.com/drive/folders/1-8RChRrXRO4F1HJv-UgaCucimihc9amy?usp=sharing) and change path in defaults.py (i.e. `_C.DATASET.png_path_mini_local = '/data/ruizhu/ORmini-pngs'`)
    - Download [other data](https://drive.google.com/drive/folders/1otm31GBHdmTTsyjbzGRqOLU4eyBwJ63s?usp=sharing) and change path in defaults.py (i.e. `_C.DATASET.dataset_path_mini_local = '/data/ruizhu/openrooms_mini'`)
    - [NOTE] To use mini for training, you need to set `DATASET.mini True` in your script
  - full dataset
    - TBD
<!-- Install Conda and create environment: -->

<!-- https://docs.conda.io/en/latest/miniconda.html -->
<!-- wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -->
<!-- bash Miniconda3-latest-Linux-x86_64.sh # choose Miniconda installation path as /{}data/miniconda3 -->

```bash
conda create -n python=3.8
conda activate py38

conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
cd /{}data
git clone https://github.com/{}/indoorInverse.git # change to your forked repo so that it's yours
cd indoorInverse/train
pip install -r requirements.txt
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 5321 --nproc_per_node=1 train/trainBRDFLight.py --task_name trainBRDF_locally --if_train True --if_val True --if_vis True --eval_every_iter 5000 --if_overfit_train False DATASET.num_workers 24 MODEL_BRDF.enable True MODEL_BRDF.load_pretrained_pth False MODEL_BRDF.enable_BRDF_decoders True MODEL_BRDF.enable_list al_ro_no_de MODEL_BRDF.loss_list al_ro_no_de DATA.data_read_list al_ro_no_de DATA.im_height 240 DATA.im_width 320 train_h 240 train_w 320 opt.cfg.DATASET.tmp False DEBUG.if_dump_perframe_BRDF True SOLVER.ims_per_batch 8 TEST.ims_per_batch 8 DATA.load_brdf_gt True DATA.if_load_png_not_hdr False DATASET.mini False MODEL_BRDF.load_pretrained_pth True
```
## 1.2 To run on the cluster
### Create your own PVC stroage as your personal ehome path

Set default Kubernates namespace:
```
kubectl config set-context --current --namespace=mc-lab
```

Got to ./cluster_control/

```bash
kubectl create -f your_pvc.yaml # repalce the {} with your name, and change the size if desired
kubectl get pvc {}data # check pvc creation result
```

### Create a persistent pod and create your conda environment
Go to ./cluster_control/

```bash
kubectl create -f your_deployment.yaml # create a persistent job (deployment) with little resources usage
kubectl get pods -ww # get the full name of yoru deployment
kubectl exec -it {}-deployment{} -- /bin/bash
mkdir tasks

```

Create conda environment:

```bash
cd /{}data
mkdir envs

conda create --prefix /{}data/envs/py38 python=3.8 # repalce the {} with your name
conda activate /{}data/envs/py38

conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
git clone https://github.com/jerrypiglet/indoorInverse.git # change to your forked repo so that it's yours
cd indoorInverse
pip install -r requirements.txt
mkdir logs && mkdir Summary_vis && mkdir Checkpoint && mkdir job_list
```

Download [pretrained checkpoints](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/models.zip) to */{}data/indoorInverse/models_ckpt*:

```bash
cd /{}data/indoorInverse
mkdir models_ckpt
cd models_ckpt
wget http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/models.zip
unzip models.zip
mv models/check* .
rm -rf models
```
### Create temporary pod and run your prototyping/debugging script within
Important params to set in your script before you launch a job:
- `--if_cluster`
- Set number of resources according to your need (e.g. `--gpus 2`, `--memr 40 --meml 70 --cpur 25 --cpul 45`). Remember to monitor the usage at the [Dashborad](https://grafana.nrp-nautilus.io/d/85a562078cdf77779eaa1add43ccec1e/kubernetes-compute-resources-namespace-pods?orgId=1&refresh=10s&var-datasource=default&var-cluster=&var-namespace=mc-lab) and address any warnings in red on your [portal homepage](https://nautilus.optiputer.net) by kill and relaunch the job with proper params
- Fill in the `{}` in *your_torch_job_mclab.yaml* with your user information.

Launch a pod with proper configurations **(with at least 1 GPU and proper CPU/memory)**:

```bash
kubectl create -f your_deployment.yaml # create a persistent job (deployment); CHNAGE THE CONFIGURATIONS (GPU etc.)!
kubectl exec -it {}-deployment{} -- /bin/bash
which python && . /root/miniconda3/etc/profile.d/conda.sh && conda activate /{}data/envs/py38 && which python # make sure the output is your conda Python
jupyter lab --no-browser --port 6006 --allow-root
```

Forward the port to your computer:

```bash
kubectl port-forward {}-deployment{} 6006:6006
```

On your browser open the link printed out from Jupyter lab (e.g. http://localhost:6006/lab?token={....}). Then you can create a terminal or browse files.

**NOTE:** Your pod will like be flagged red due to low resource usage, and you will need to terminate your current pods to launch new ones after two on-going violations.

Launch a testing job within the terminal of Jupyter Lab:
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 5321 --nproc_per_node=1 train/trainBRDFLight.py --task_name testBRDF_locally --if_cluster --if_train False --if_val False --if_vis True --eval_every_iter 5000 --if_overfit_train False DATASET.num_workers 24 MODEL_BRDF.enable True MODEL_BRDF.load_pretrained_pth False MODEL_BRDF.enable_BRDF_decoders True MODEL_BRDF.enable_list al_ro_no_de MODEL_BRDF.loss_list al_ro_no_de DATA.data_read_list al_ro_no_de DATA.im_height 240 DATA.im_width 320 train_h 240 train_w 320 opt.cfg.DATASET.tmp False DEBUG.if_dump_perframe_BRDF True SOLVER.ims_per_batch 8 TEST.ims_per_batch 8 DATA.load_brdf_gt True DATA.if_load_png_not_hdr False DATASET.mini False MODEL_BRDF.load_pretrained_pth True
```

The logged Tensorboard files will be saved to `/logs` of the root path of the repo. You can launch a tensorboard instance within the terminal and forward to your local machine to inspect the results.

### Submit a job for training/batch evaluating
```bash
(py38) ruizhu@ubuntu:~/Documents/Projects/indoorInverse/cluster_control$ python your_tool.py create -d --gpus 2 -f your_torch_job_mclab.yaml --memr 40 --meml 70 --cpur 25 --cpul 45 -s 'python -m torch.distributed.launch --master_port 5320 --nproc_per_node=2  trainBRDFLight.py --if_cluster --task_name DATE-train_POD_trainBRDF_scratch --if_train True --if_val True --if_vis True --eval_every_iter 5000 --if_overfit_train False DATASET.num_workers 24 MODEL_BRDF.enable True MODEL_BRDF.load_pretrained_pth False MODEL_BRDF.enable_BRDF_decoders True MODEL_BRDF.enable_list al_de_no_ro MODEL_BRDF.loss_list al_de_no_ro DATA.data_read_list al_de_no_ro DATA.im_height 240 DATA.im_width 320 train_h 240 train_w 320 opt.cfg.DATASET.tmp False DEBUG.if_dump_perframe_BRDF True SOLVER.ims_per_batch 8 TEST.ims_per_batch 8 DATA.load_brdf_gt True DATA.if_load_png_not_hdr False DATASET.mini False MODEL_BRDF.load_pretrained_pth True'
```

Common commands to minitor your jobs (fill in `{}` with your pod/deployment name):
- kubectl get pods -ww
- kubectl get deployments -ww
- kubectl delete {pod name}
- kubectl logs -f {pod name}
- kubectl describe pods {pod name}
- kubectl exec -it {pod name} -- /bin/bash
- kubectl delete deployments {deployments name}

## 1.3 Notes by Rui
Not ready:

- cascade > 1
- test on real images (real images, IIW, nyud)
- lighting data on the cluster and data loading of it
  
Training script used by Rui:
```bash
python rui_tool.py create -d --gpus 1 -f rui_torch_job_2gpu_v6_mclab.yaml --memr 20 --meml 50 --cpur 15 --cpul 30 -s 'python -m torch.distributed.launch --master_port 5320 --nproc_per_node=1 trainBRDFLight.py --if_cluster --task_name DATE-train_POD_trainBRDF_scratch --if_train True --if_val True --if_vis True --eval_every_iter 5000 --if_overfit_train False DATASET.num_workers 14 MODEL_BRDF.enable True MODEL_BRDF.load_pretrained_pth False MODEL_BRDF.enable_BRDF_decoders True MODEL_BRDF.enable_list al_de_no_ro MODEL_BRDF.loss_list al_de_no_ro DATA.data_read_list al_de_no_ro DATA.im_height 240 DATA.im_width 320 train_h 240 train_w 320 opt.cfg.DATASET.tmp False DEBUG.if_dump_perframe_BRDF True SOLVER.ims_per_batch 8 TEST.ims_per_batch 8 DATA.load_brdf_gt True DATA.if_load_png_not_hdr False DATASET.mini False MODEL_BRDF.load_pretrained_pth False'
```

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
