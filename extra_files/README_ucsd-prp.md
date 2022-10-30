## 1.2 To run on the UCSD-prp cluster
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
mkdir logs && mkdir summary_vis && mkdir Checkpoint && mkdir job_list
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
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 5321 --nproc_per_node=1 train/trainBRDFLight.py --task_name testBRDF_locally --if_cluster --if_train False --if_val False --if_vis True --eval_every_iter 5000 --if_overfit_train False DATASET.num_workers 16 MODEL_BRDF.enable True MODEL_BRDF.load_pretrained_pth False MODEL_BRDF.enable_BRDF_decoders True MODEL_BRDF.enable_list al_ro_no_de MODEL_BRDF.loss_list al_ro_no_de DATA.data_read_list al_ro_no_de DATA.im_height 240 DATA.im_width 320 train_h 240 train_w 320 opt.cfg.DATASET.tmp False DEBUG.if_dump_perframe_BRDF True SOLVER.ims_per_batch 8 TEST.ims_per_batch 8 DATA.if_load_png_not_hdr True DATASET.mini False MODEL_BRDF.load_pretrained_pth False
```

The logged Tensorboard files will be saved to `/logs` of the root path of the repo. You can launch a tensorboard instance within the terminal and forward to your local machine to inspect the results.

### Submit a job for training/batch evaluating
```bash
(py38) ruizhu@ubuntu:~/Documents/Projects/indoorInverse/cluster_control$ python your_tool.py create -d --gpus 2 -f your_torch_job_mclab.yaml --memr 10 --meml 30 --cpur 5 --cpul 20 -s 'python -m torch.distributed.launch --master_port 5320 --nproc_per_node=2  trainBRDFLight.py --if_cluster --task_name DATE-train_POD_trainBRDF_scratch --if_train True --if_val True --if_vis True --eval_every_iter 5000 --if_overfit_train False DATASET.num_workers 16 MODEL_BRDF.enable True MODEL_BRDF.load_pretrained_pth False MODEL_BRDF.enable_BRDF_decoders True MODEL_BRDF.enable_list al_de_no_ro MODEL_BRDF.loss_list al_de_no_ro DATA.data_read_list al_de_no_ro DATA.im_height 240 DATA.im_width 320 SOLVER.ims_per_batch 8 TEST.ims_per_batch 8 DATA.if_load_png_not_hdr True DATASET.mini False MODEL_BRDF.load_pretrained_pth False'
```

Common commands to minitor your jobs (fill in `{}` with your pod/deployment name):
- kubectl get pods -ww
- kubectl get deployments -ww
- kubectl delete {pod name}
- kubectl logs -f {pod name}
- kubectl describe pods {pod name}
- kubectl exec -it {pod name} -- /bin/bash
- kubectl delete deployments {deployments name}
