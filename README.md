# Auto-FERNet
This is an implementation of the paper ["Auto-FERNet: A Facial Expression Recognition Network With Architecture Search"](https://ieeexplore.ieee.org/abstract/document/9442348).

Deep convolutional neural networks have achieved great success in facial expression datasets both under laboratory conditions and in the wild. However, most of these related researches use general image classification networks (e.g., VGG, GoogLeNet) as backbones, which leads to inadaptability while applying to Facial Expression Recognition (FER) task, especially those in the wild. In the meantime, these manually designed networks usually have large parameter size. To tackle with these problems, we propose an appropriative and lightweight Facial Expression Recognition Network Auto-FERNet, which is automatically searched by a differentiable Neural Architecture Search (NAS) model directly on FER dataset. Furthermore, for FER datasets in the wild, we design a simple yet effective relabeling method based on Facial Expression Similarity (FES) to alleviate the uncertainty problem caused by natural factors and the subjectivity of annotators. Experiments have shown the effectiveness of the searched Auto-FERNet on FER task. Concretely, our architecture achieves a test accuracy of 73.78% on FER2013 without ensemble or extra training data. And noteworthily, experimental results on CK+ and JAFFE outperform the state-of-the-art with an accuracy of 98.89% (10 folds) and 97.14%, respectively, which also validate the robustness of our system.

![overview](images/overview.png)Pipeline of Auto-FERNet | ![FES](images/FES.jpg)Facial Expression Similarity (FES) Matrix
---|---


## Dataloader
The dataloaders of _FER2013, CK48_ and _JAFFE_ are conducted in `dataloader/dataload_h5.py`.

## Architecture Search
We conduct Neural Achitecture Search (NAS) on _FER2013_ based on [SGAS](https://arxiv.org/abs/1912.00195).

* To search for the best architecture for _FER2013_, run
```angular2html
python train_search.py --batch_size 24 --batch_increase 8 --learning_rate 0.025
```
* The `NAS/model_search.py` and `NAS/operations.py` defines the search space and the candidate operations in each edge.
* The `NAS/architect.py` consists of the computational steps in architecture search.
* The `NAS/genotypes.py` saves the searched architecture. You can add the searched structure into this file manually.

## Retraining on _FER2013_
After search the optimal architecture on _FER2013_, you can retrain the model from scratch and test it by running:
```angular2html
python train_FER2013.py --batch_size 24 --learning_rate 0.025 --layers 12 --auxiliary_weight 0.4
```

## Relabeling
To further improve the performance on _FER2013_ by reducing uncertainty and executing a robust training, load the saved model after retraining and run:
```angular2html
python train_relabel.py --batch_size 64 --learning_rate 0.02272721 --relabel_threshold 0.2 --fes True --fes_threshold 0.03
```

Note that the loaded models are, ideally, the ones saved before totally converged, such as when the training accuracy reaches _85%_ or _90%_. 

The relabeling can be divided into two strategies.
1. The original relabeling based only on the softmax vector
2. The relabeling based on Facial Expression Similarity (FES)

You are free to customize your relabeling strategy by varing the args (relabel_threshold, fes and fes_threshold) in `train_relabel.py`.

## Ensemble
To get an average inference from different models on _FER2013_. You can customize your ensemble models in the array of `model_names` and `--layers` and run:
```angular2html
python ensemble.py --batch_size 64
```

## Fine-tuning on _CK48_ and _JAFFE_
After retraining on _FER2013_, you can fine-tune the saved model on _CK48_ and _JAFFE_ by runing:
```angular2html
python train_CK48.py --batch_size 64 --learning_rate 0.01794073
```
```angular2html
python train_CK48.py --batch_size 16 --learning_rate 0.01794073
```

## Other Tools
More tools for analysis and visualization are involved in the `tools` folder.

## Results
### The searched cells (normal cell and reduction cell):

<div align="center"><img alt="cells.png" width=500 height=600 src="images/cells.png"/></div>

### Performance

<!-- <div align="center"><img alt="FER2013.png" src="images/FER2013.png"/></div>
<div align="center"><img alt="CK_JAFFE.png" src="images/CK_JAFFE.png"/></div> -->

|Benchmark|Params(MB)|Accuracy|
|:---:|:---:|:---:|
| FER2013|2.1|73.78/74.98 (6 ensemble)|
|CK48|2.1|99.37 (8 folds)/98.89 (10 folds)|
|JAFFE|2.1|97.14|

## Citation
If you think our work inspires you, please cite our paper in your work.

Plain Text:
>S. Li et al., "Auto-FERNet: A Facial Expression Recognition Network With Architecture Search," in IEEE Transactions on Network Science and Engineering, vol. 8, no. 3, pp. 2213-2222, 1 July-Sept. 2021, doi: 10.1109/TNSE.2021.3083739.

BibTex:
>@ARTICLE{9442348,  
> author={Li, Shiqian and Li, Wei and Wen, Shiping and Shi, Kaibo and Yang, Yin and Zhou, Pan and Huang, Tingwen},  
> journal={IEEE Transactions on Network Science and Engineering},   
> title={Auto-FERNet: A Facial Expression Recognition Network With Architecture Search},   
> year={2021},  
> volume={8},  
> number={3},  
> pages={2213-2222},  
> doi={10.1109/TNSE.2021.3083739}}
