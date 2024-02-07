# BiasedAD

## environment

Python version 3.7.16.

Create suitable conda environment:

```
conda env create -f environment.yml
```


## fashionMNIST

For the fashionMNIST dataset, our training data contains three categories: normal, non_target, and target, which need to be explicitly specified in the sh command.

**$\text{BiasedAD}$**

```sh
exec -a "BAD_420" python main.py --model_type BiasedAD --dir_path ./result/fmnist --dataset_name fashionmnist --normal_class 4 --non_target_outlier_class 2 --target_outlier_class 0 --gpu 2 --random_seed 0&
```

**$\text{BiasedAD}^\text{M}$**

$\text{BiasedAD}^\text{M}$ is similar to $\text{BiasedAD}$. Although the command includes ``--non_target_outlier_class``, the number of non-target anomalies is set to 0 during runtime. The required sampling count is set to default 100 and does not need to be explicitly declared.

```sh
exec -a "BADM_420" python main.py --model_type BiasedADM --dir_path ./result/fmnist --dataset_name fashionmnist --normal_class 4 --non_target_outlier_class 2 --target_outlier_class 0 --gpu 2 --random_seed 0 &
```

## nb-15

**$\text{BiasedAD}$**

```sh
exec -a "BAD_nb15" python main.py --model_type BiasedAD --dir_path ./result/nb15 --dataset_name nb15 --gpu 2 --random_seed 0&
```

**$\text{BiasedAD}^\text{M}$**

$\text{BiasedAD}^\text{M}$ adds the parameter ``--sample_count`` compared to $\text{BiasedAD}$. Since the default sampling count is 100, which is specific to the fashionMNIST dataset, it's necessary to declare ``--sample_count 1000``

```sh
exec -a "BADM_nb15" python main.py --model_type BiasedADM --dir_path ./result/nb15 --dataset_name nb15 --gpu 0 --sample_count 1000 --random_seed 0 &
```
