# Auto-FERNet
This is an implementation of the paper ["Auto-FERNet: A Facial Expression Recognition Network With Architecture Search"](https://ieeexplore.ieee.org/abstract/document/9442348).
<div align="center"><img alt="overview.png" src="images/overview.png"/></div>

## Dataloader
The dataloaders of _FER2013, CK48_ and _JAFFE_ are conducted in `dataloader/dataload_h5.py`.

## Architecture Search
We conduct Neural Achitecture Search (NAS) on _FER2013_ based on [SGAS](https://arxiv.org/abs/1912.00195).

* Run `train_search.py` to search for the best architecture for _FER2013_.
* The `NAS/model_search.py` and `NAS/operations.py` defines the search space and the candidate operations in each edge.
* The `NAS/architect.py` consists of the computational steps in architecture search.
* The `NAS/genotypes.py` saves the searched architecture. You can add the searched structure into this file manually.

## Retraining on _FER2013_
After search the optimal architecture on _FER2013_, you can run `train_FER2013.py` to retrain the model from scratch and test it.

## Relabeling
To further improve the performance on _FER2013_ by reducing uncertainty and executing a robust training, load the saved model after retraining and run `train_relabel.py`. Note that the loaded models are, ideally, the ones saved before totally converged, such as when the training accuracy reaches _85%_ or _90%_. 

The relabeling can be divided into two strategies.
1. The original relabeling based only on the softmax vector
2. The relabeling based on Facial Expression Similarity (FES)

You are free to customize your relabeling strategy by varing the args (relabel_threshold, fes and fes_threshold) in `train_relabel.py`.

## Ensemble
Run `ensemble.py` to get an average inference from different models on _FER2013_. You can customize your ensemble models in the array of `model_names`.

## Fine-tuning on _CK48_ and _JAFFE_
After retraining on _FER2013_, you can fine-tune the saved model on _CK48_ and _JAFFE_ by runing `train_CK48.py` and `train_JAFFE.py`.

## Other Tools
More tools for analysis and visualization are involved in the `tools` folder.

## Results
### The searched cells (normal cell and reduction cell):

<img alt="cells.png" src="images/cells.png"/>

### Performance

<div align="center"><img alt="FER2013.png" src="images/FER2013.png"/></div>
<div align="center"><img alt="CK_JAFFE.png" src="images/CK_JAFFE.png"/></div>

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