# FrictionEstimation
A Tire-Road Friction Coefficient estimation method based on camera-informing and dynamic analysis.

## 1. Camera-based Informing

This part gives a classification result of the surface by analyzing the texture of the surface through GLCM((Gray Level Co-occurrence Matrix)) and extracting fueature by pretrained ResNet. Later we will use this classification to inform the dynamic model.

### 1.1 NetWork Architecture

![ClassificationArchitecture](illustrations/ClassificationArchitecture.png)

### 1.2 Files

-  `/data`: ground truth videos, classified images(training, dev and test data);
-  `/data/dataset.py`: Pytorch Dataset Class, used for Dataloader;
-  `/data/data_preprocess.py`: Convert videos to imgs and split into train, dev and test sets;
-  `/model`: trained models and checkpoints;
-  `train.py`: main training script;
-  `evaluate.py`: run model on test set to evaluate it;
-  `model.py`: definition of the NN and the layers;
-  `util.py`: helper functions, like the implementation of glcm.

### 1.3 Commands

To train the model, run (set up the WANDB_API_KEY in util.py if using wandb):

```
python3 train.py --use_wandb --experiment_name="your-experiment-name" --batch_size=40 --lr=0.001 --lr_decay=0.1 --max_epoch=40 --num_workers=12 --drop_out=0.1
```
To evaluate on test, run:
```
python3 evaluate.py
```

### 1.4 Current Performance

- 06.23 Using ResNet50

![Alt text](illustrations/0623-test-result.png)

- 06.30 Using ResNet18

![Alt text](illustrations/0629-Res18-result.png)

- 06.30 Using TensorScript for Acceleration

![Alt text](illustrations/0630-Res18-TensorScript.png)


meanshift

fundation model?

DBSTREAM（Density-Based Stream Clustering）是一个用于处理数据流的增量密度聚类算法。它能够处理不断到来的数据流，并动态调整簇的结构。以下是DBSTREAM的关键参数及其作用的详细解释：

参数分析
clustering_threshold:

作用：决定微簇的创建和合并阈值。该参数定义了在创建新微簇或合并现有微簇时的距离阈值。
增大：增大该值会导致更少的微簇被创建，因为距离较远的数据点仍可能被认为属于同一个微簇。结果是更大的簇，较少的簇数。
减小：减小该值会导致更多的微簇被创建，因为距离较近的数据点才会被认为属于同一个微簇。结果是更多的小簇。
fading_factor:

作用：控制微簇的衰减速率。这个参数决定了微簇随时间的权重递减速度。
增大：增大该值会加速微簇的衰减，使得较老的数据点对当前聚类结果的影响减少。结果是聚类更多地反映最近的数据。
减小：减小该值会减缓微簇的衰减，使得较老的数据点对当前聚类结果的影响增加。结果是聚类反映的时间范围更广，历史数据的影响较大。
cleanup_interval:

作用：定义清理无效微簇的时间间隔。无效微簇是那些权重低于某一阈值的微簇。
增大：增大该值会减少清理操作的频率，这可能会导致更多的无效微簇存在，增加计算量。
减小：减小该值会增加清理操作的频率，及时删除无效微簇，减少计算量，但可能影响性能。
intersection_factor:

作用：控制微簇合并的判定条件。该参数决定了两个微簇在何种程度上的重叠会导致它们合并。
增大：增大该值会使得两个微簇更容易被合并。结果是更大的簇，较少的簇数。
减小：减小该值会使得两个微簇更难被合并。结果是更多的小簇，较多的簇数。
minimum_weight:

作用：定义微簇的最小权重。低于此权重的微簇会被清理掉。
增大：增大该值会导致更多的微簇被清理，因为更多的微簇会低于这个权重阈值。结果是减少了簇的数量。
减小：减小该值会保留更多的微簇，因为更少的微簇会低于这个权重阈值。结果是增加了簇的数量。
参数调整的影响
clustering_threshold 和 intersection_factor 控制簇的大小和数量。增大它们会导致更少、更大的簇；减小它们会导致更多、更小的簇。
fading_factor 和 cleanup_interval 控制微簇的动态性和历史数据的影响。增大fading_factor和减小cleanup_interval会使聚类更关注最近的数据；减小fading_factor和增大cleanup_interval会使聚类更关注历史数据。
minimum_weight 控制微簇的有效性和数量。增大它会减少簇的数量，因为更多的微簇会被清理；减小它会增加簇的数量，因为更多的微簇会被保留。
这些参数的选择应根据具体的数据特性和应用场景进行调整，以实现最佳的聚类效果。希望这些解释能帮助你更好地理解和调整DBSTREAM的参数设置。