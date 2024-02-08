# GAD

## environment

Python version 3.7.16.

Create suitable conda environment:

```
conda env create -f environment.yml
```


## fashionMNIST

For the fashionMNIST dataset, our training data contains three categories: normal, non_target, and target, which need to be explicitly specified in the sh command.

**$\text{GAD}^{f-partial}$**

```sh
python main.py --model_type GADF --dir_path ./result/fmnist --dataset_name fashionmnist --normal_class 4 --non_target_outlier_class 2 --target_outlier_class 0 --gpu 2 --random_seed 0&
```

**$\text{GAD}^{s-partial}$**

$\text{GAD}^{s-partial}$ is similar to $\text{GAD}^{f-partial}$. Although the command includes ``--non_target_outlier_class``, the number of non-target anomalies is set to 0 during runtime. The required sampling count is set to default 100 and does not need to be explicitly declared.

```sh
python main.py --model_type GADS --dir_path ./result/fmnist --dataset_name fashionmnist --normal_class 4 --non_target_outlier_class 2 --target_outlier_class 0 --gpu 2 --random_seed 0 &
```

**$\text{GAD}^{con}$**

The code for $\text{GAD}^{con}$ is identical to that of $\text{GAD}^{s-partial}$, but it utilizes data from conventional anomaly detection tasks.

```sh
python main.py --model_type GADS --dir_path ./result/fmnist --dataset_name fashionmnist --normal_class 4 --non_target_outlier_class 4 --target_outlier_class 6 --gpu 2 --random_seed 0 &
```

## nb-15

**$\text{GAD}^{f-partial}$**

```sh
python main.py --model_type GADF --dir_path ./result/nb15 --dataset_name nb15 --gpu 2 --random_seed 0&
```

**$\text{GAD}^{s-partial}$**

$\text{GAD}^{s-partial}$ adds the parameter ``--sample_count`` compared to $\text{GAD}^{f-partial}$. Since the default sampling count is 100, which is specific to the fashionMNIST dataset, it's necessary to declare ``--sample_count 1000``

```sh
python main.py --model_type GADS --dir_path ./result/nb15 --dataset_name nb15 --gpu 0 --sample_count 1000 --random_seed 0 &
```
