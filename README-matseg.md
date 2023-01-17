Train:

``` bash
CUDA_VISIBLE_DEVICES=1 python train/trainMatseg.py --task_name matseg_OR --if_train True --if_eval True --if_vis True --config openrooms_train_matseg.yaml
CUDA_VISIBLE_DEVICES=1 python train/trainMatseg.py --task_name matseg_OR_D16 --if_train True --if_eval True --if_vis True --config openrooms_train_matseg.yaml MODEL_MATSEG.embed_dims 16
```

Vis on train images:

``` bash
CUDA_VISIBLE_DEVICES=0 python train/trainMatseg.py --task_name matseg-OR-testOnTrain-resume20230115-180728 --if_train False --if_eval False --if_vis True --if_overfit_train True --config openrooms_train_matseg.yaml --resume 20230115-180728--matseg_OR
```

Test on Indoor-kitchen:

``` bash
CUDA_VISIBLE_DEVICES=0 python train/trainMatseg.py --task_name tmp --if_train False --if_eval False --if_vis True --if_overfit_train True --config Indoor_test_matseg.yaml --resume 20230115-180728--matseg_OR --test_real
```