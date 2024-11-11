# GNN-GAN
Pytorch implementation of paper ['GNN-GAN for Node Classification on Class-imbalanced Graph Data'](https://arxiv.org/abs/2103.08826) on WSDM2021

## Dependencies
### CPU
- python3
- ipdb
- pytorch1.0
- network 2.4
- scipy
- sklearn

## Dataset
Three processed datasets are published, including Cora,Citeseer and BlogCatalog. They have been provided in Data folder.

## Configurations
Please find details in utils.py

### Architectures
We provide three base architectures, GAT, GCN and GraphSage. The default one is GAT, and can be set via '--model'.

### Upscale ratios
The default value is 1. If you want to make every class balanced instead of using pre-set ratios, please set it to 0 in '--up_scale'.

## GNN-GAN
Below is an example for the Cora dataset.

### Train
- Pretrain the auto-encoder

<code>python main.py --imbalance --no-cuda --dataset=cora --setting='recon'</code>


### Baselines
We provide four baselines in this code. They can be configured via the '--setting' arguments. Please refer to the 'get_parser()' function in utils.py.
- Oringinal model: Vanilla backbone models. '--setting='no''
- Over-sampling in raw input domain: Repeat nodes in the minority classes. '--setting='upsampling''
- SMOTE: Perform SMOTE in the intermediate embedding domain. '--setting='smote''
- GAN: perform GNN-GAN. '--setting='GAN' '
- GAN: performan GNN+GAN '--setting='no' ' and the datasets should be '--dataset=='cora_new''


## Citation

If any problems occur via running this code, please contact us at 20224046003@stu.suda.edu.cn.

Thank you!


