# FrictionEstimation
A Tire-Road Friction Coefficient estimation method based on camera-informing and dynamic analysis.

## 1. Camera-based Informing

This part gives a classification result of the surface by analyzing the texture of the surface through GLCM((Gray Level Co-occurrence Matrix)) and extracting fueature by pretrained ResNet. Later we will use this classification to inform the dynamic model.

### 1.1 NetWork Architecture

![image.png](https://s2.loli.net/2024/09/11/yjX2YbiTNFU3Cmk.png)

### 1.2 Files

-  `/data`: ground truth videos, classified images(training and test data);
-  `/data/dataset.py`: Pytorch Dataset Class, used for Dataloader;
-  `/model`: trained models and checkpoints;
-  `train-GLCM.py`: main training script;
-  `model.py`: definition of the NN and the layers;
-  `util.py`: helper functions, like the implementation of glcm.

### 1.3 Commands

To train the model, run (set up the WANDB_API_KEY in util.py if using wandb):

```
python .\train-GLCM.py --num_workers 12 --batch_size 128 --max_epoch 100 --lr 0.001 --lr_decay 0.5 --patience 5 --experiment_name "your-experiment-name" --use_wandb
```

### 1.4 Dataset Information

| Label | Surface          | Sample |
|-------|------------------| --- |
| 0     | 6edge_gray_brick |![image_0189.png](https://s2.loli.net/2024/09/11/FV4T5yRkrd9UPcZ.png) |
| 1     | cement_pavement  |![image_0751.png](https://s2.loli.net/2024/09/11/Nz4ZrKGtdcYHiQ9.png)|
| 2     | grass            |![image_0012.png](https://s2.loli.net/2024/09/11/qarpRl6JXACOd79.png)|
| 3     | levin_between    |![levin_between_frame_0485.jpg](https://s2.loli.net/2024/09/11/rEXvRkj71itJAb2.jpg)|
| 4     | levin_carpet     |![levin_carpet_frame_0189.jpg](https://s2.loli.net/2024/09/11/DXiujq5pCePb1QZ.jpg)|
| 5     | levin_desk       |![levin_desk_frame_0150.jpg](https://s2.loli.net/2024/09/11/Zt81RkHVXC4DxS6.jpg)|
| 6     | levin_hall       |![image_0064.png](https://s2.loli.net/2024/09/11/4oaHSvh7zgdFlmO.png)|
| 7     | levin_tape       |![levin_tape_frame_0076.jpg](https://s2.loli.net/2024/09/11/y17KBdeCYsxT3n8.jpg)|
| 8     | levin_bricks     |![bricks_frame_0199.jpg](https://s2.loli.net/2024/09/11/PGeJhZ8jmOSLvbf.jpg)|
| 9     |moore_elevator|![elevator_frame_0180.jpg](https://s2.loli.net/2024/09/11/Gotk3OgCAqKQSNY.jpg)
| 10 | moore_Klab | ![moore_frame_0387.jpg](https://s2.loli.net/2024/09/11/lHF2fO6e3hkoXvZ.jpg) |
| 11 | race_track_brick | ![race_track3_frame_1297.jpg](https://s2.loli.net/2024/09/11/MXytdwfYOqIGePK.jpg) |
| 12 | red_brick | ![image_0666.png](https://s2.loli.net/2024/09/11/p6K9wUeJbDVhoHP.png) |
| 13 | wood_bridge | ![image_1261.png](https://s2.loli.net/2024/09/11/n89KHCc3GaUpSv5.png) |
