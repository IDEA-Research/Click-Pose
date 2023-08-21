# Neural Interactive Keypoint Detection

<img src="assets/main_clickpose.jpg" />  

[Jie Yang](https://github.com/yangjie-cv), [Ailing Zeng](https://ailingzeng.site/), [Feng Li](https://scholar.google.com/citations?user=ybRe9GcAAAAJ&hl=zh-CN), [Shilong Liu](http://www.lsl.zone/), [Ruimao Zhang](http://www.zhangruimao.site/), [Lei Zhang](https://www.leizhang.org/)

This is the official pytorch implementation of our ICCV 2023 paper "Neural Interactive Keypoint Detection". 

## 💡 Click-Pose

<img src="assets/framework_clickpose.jpg" />  

- #### We first propose interactive keypoint detection task towards efficient keypoint annotation.

- #### We present the first neural interactive keypoint detection framework, Click-Pose, as a baseline for further research. 
## ▶ Demo

#### Operation Flow: 🤖 Model localizes all keypoints ➡ 👨 User corrects a few wrong keypoints ➡ 🤖 Model refines other keypoints


#### In-Domain Annotation
<img src="assets/In-domain.gif" />  

```
Given a natural multi-person scene, we first utilize our model first obtains the predicted keypoints and human boxes. By setting a classification score threshold to 0.1, we can obtain these candidate detections. Here, the candidate poses are sorted in descending order of their classification scores. We check the first candidate. Its keypoints and human boxes are almost correct. For the second one, we can see that the network predicted this person’s pose in reverse. This should be the person's right side, but the model predicts it as the left side. Thus, we provide feedback by correcting one keypoint, such as moving the left elbow to the correct position, and then feed this modified pose back to our model. As a result, our model can output the refined predicted keypoints and human boxes. We check the new candidate detections. We first notice that our model changes the classification scores of the previous predictions. By setting a classification score threshold to 0.1, we can filter out more inaccurate candidates. And now, the person candidate that we corrected has the highest classification score. Then, we can observe that for this person, the network realizes the flipping error based on our modification of the left elbow position, and subsequently refines the positions of all other keypoints, resulting in an accurate pose.
```


#### Out-of-Domain Annotation
<img src="assets/Out-of-domain.gif" />  

```
Given an out-of-domain artificial scene, we also obtain the predicted pose and human boxes. However, since this is out-of-domain data, the model cannot provide an accurate pose. We correct the left ankle to the correct position and use the model to refine the predictions. We observe that the network now outputs the correct pose and human box.
```



## 🚀 Model Zoo

### 1. Model-Only Results 

#### COCO val2017 set

|   Model    | Backbone  | Lr schd | mAP  | AP<sup>50</sup> | AP<sup>75</sup> | AP<sup>M</sup> | AP<sup>L</sup> | Time (ms) |                                                 Model                                                 |
|:----------:|:---------:|:-------:|:----:|:---------------:|:---------------:|:--------------:|:--------------:|:---------:|:-----------------------------------------------------------------------------------------------------:|
|  ED-Pose   | ResNet-50 |   60e   | 71.7 |      89.7       |      78.8       |      66.2      |      79.7      |    51     |                          [GitHub](https://github.com/IDEA-Research/ED-Pose)                           |
| Click-Pose |   ResNet-50    |   40e   | 73.0 |      90.4       |      80.0       |      68.1      |      80.5      |    48     | [Google Drive](https://drive.google.com/file/d/1_rp12m0fkpSc7LQ1oXeifdt8SbwcSHtS/view?usp=sharing) |

#### Human-Art val set

|   Model    |   Backbone     | mAP  | AP<sup>M</sup> | AP<sup>L</sup> |                                           Model                                                |
|:----------:|:-------------:|:----:|:--------------:|:--------------:|:-----------------------------------------------------------------------------------------------------:|
|  ED-Pose   |     ResNet-50        | 37.5 |      7.6       |      41.1      |     [GitHub](https://github.com/IDEA-Research/ED-Pose)      |
| Click-Pose |    ResNet-50       | 40.5 |      8.3       |      44.2      | [Google Drive](https://drive.google.com/file/d/1_rp12m0fkpSc7LQ1oXeifdt8SbwcSHtS/view?usp=sharing) |

#### OCHuman test set

|   Model    |   Backbone     | mAP  | AP<sup>50</sup> | AP<sup>75</sup> |                                           Model                                                |
|:----------:|:-------------:|:----:|:---------------:|:---------------:|:-----------------------------------------------------------------------------------------------------:|
|  ED-Pose   |     ResNet-50        | 31.4 |      39.5       |      35.1       |     [GitHub](https://github.com/IDEA-Research/ED-Pose)      |
| Click-Pose |    ResNet-50       | 33.9 |      43.4       |      37.5       | [Google Drive](https://drive.google.com/file/d/1_rp12m0fkpSc7LQ1oXeifdt8SbwcSHtS/view?usp=sharing) |

Note that the model is trained on COCO train2017 set and tested on COCO val2017 set, Human-Art val set, and OCHuman test set.

### 2. Neural Interactive  Results 

#### In-domain Annotation (COCO val2017)

|   Model    |   Backbone     | NoC@85 | NoC@90 | NoC@95 |                                           Model                                                |
|:----------:|:-------------:|:------:|:------:|:------:|:-----------------------------------------------------------------------------------------------------:|
|  ViTPose   |     ResNet-50        |  1.46  |  2.15  |  2.87  |     [GitHub](https://github.com/ViTAE-Transformer/ViTPose)      |
| Click-Pose |    ResNet-50       |  0.95  |  1.48  |  1.97  | [Google Drive](https://drive.google.com/file/d/184RIVxFVrDho4Nw5Yquh6fedTKpsZVYX/view?usp=sharing) |



#### Out-of-domain Annotation (Human-Art val)

|   Model    |   Backbone     | NoC@85 | NoC@90 | NoC@95 |                                           Model                                                |
|:----------:|:-------------:|:------:|:------:|:------:|:-----------------------------------------------------------------------------------------------------:|
|  ViTPose   |     ResNet-50        |  9.12  |  9.79  | 10.13  |     [GitHub](https://github.com/ViTAE-Transformer/ViTPose)      |
| Click-Pose |    ResNet-50       |  4.82  |  5.81  |  6.45  | [Google Drive](https://drive.google.com/file/d/184RIVxFVrDho4Nw5Yquh6fedTKpsZVYX/view?usp=sharing) |


## 🔨 Environment Setup 

<details>
  <summary>Installation</summary>
  
  We use the [ED-Pose](https://github.com/IDEA-Research/ED-Pose) as our codebase. We test our models under ```python=3.7.3,pytorch=1.9.0,cuda=11.1```. Other versions might be available as well.

   1. Clone this repo
   ```sh
   git clone https://github.com/IDEA-Research/Click-Pose.git
   cd Click-Pose
   ```

   2. Install Pytorch and torchvision

   Follow the instruction on https://pytorch.org/get-started/locally/.
   ```sh
   # an example:
   conda install -c pytorch pytorch torchvision
   ```

   3. Install other needed packages
   ```sh
   pip install -r requirements.txt
   ```

   4. Compiling CUDA operators
   ```sh
   cd models/clickpose/ops
   python setup.py build install
   # unit test (should see all checking is True)
   python test.py
   cd ../../..
   ```
</details>

<details>
  <summary>Data Preparation</summary>

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download). 
The coco_dir should look like this:
```
|-- Click-Pose
`-- |-- coco_dir
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```
**For Human-Art data**, please download from [Human-Art download](https://github.com/IDEA-Research/HumanArt), 
The humanart_dir should look like this:
```
|-- Click-Pose
`-- |-- humanart_dir
    `-- |-- annotations 
        |   |-- training_humanart.json
        |   |-- validation_humanart.json
        `-- images
            |-- 2D_virtual_human
                |-- ...
            |-- 3D_virtual_human
                |-- ...
            |-- real_human
                |-- ...
```


**For CrowdPose data**, please download from [CrowdPose download](https://github.com/Jeff-sjtu/CrowdPose#dataset), 
The crowdpose_dir should look like this:
```
|-- Click-Pose
`-- |-- crowdpose_dir
    `-- |-- json
        |   |-- crowdpose_train.json
        |   |-- crowdpose_val.json
        |   |-- crowdpose_trainval.json (generated by util/crowdpose_concat_train_val.py)
        |   `-- crowdpose_test.json
        `-- images
            |-- 100000.jpg
            |-- 100001.jpg
            |-- 100002.jpg
            |-- 100003.jpg
            |-- 100004.jpg
            |-- 100005.jpg
            |-- ... 
```
**For OCHuman data**, please download from [OCHuman download](https://github.com/liruilong940607/OCHumanApi). 
The ochuman_dir should look like this:
```
|-- Click-Pose
`-- |-- ochuman_dir
    `-- |-- annotations
        `-- images
```

</details>


## 🥳 Run


### Train on COCO:

<details>
  <summary>Model-Only</summary>

```
export CLICKPOSE_COCO_PATH=/path/to/your/coco_dir
 python -m torch.distributed.launch --nproc_per_node=4 main.py \
    --output_dir "logs/ClickPose_Model-Only" \
    -c config/clickpose.cfg.py \
    --options batch_size=4 epochs=100 lr_drop=80 use_ema=TRUE human_feedback=FLASE feedback_loop_NOC_test=FALSE feedback_inference=FALSE only_correction=FALSE
    --dataset_file="coco"
```
</details>


<details>
  <summary>Neural Interactive</summary>

```
export CLICKPOSE_COCO_PATH=/path/to/your/coco_dir
 python -m torch.distributed.launch --nproc_per_node=4 main.py \
    --output_dir "logs/ClickPose_Neural_Interactive" \
    -c config/clickpose.cfg.py \
    --options batch_size=4 epochs=100 lr_drop=80 use_ema=TRUE human_feedback=TRUE feedback_loop_NOC_test=FALSE feedback_inference=FALSE only_correction=FALSE
    --dataset_file="coco"
```
</details>



### Evaluation on COCO:

<details>
  <summary>Model-Only</summary>

```
export CLICKPOSE_COCO_PATH=/path/to/your/coco_dir
 python -m torch.distributed.launch --nproc_per_node=4 main.py \
    --output_dir "logs/ClickPose_Model-Only_eval" \
    -c config/clickpose.cfg.py \
    --options batch_size=4 epochs=100 lr_drop=80 use_ema=TRUE human_feedback=FLASE feedback_loop_NOC_test=FALSE feedback_inference=FALSE only_correction=FALSE
    --dataset_file="coco"
    --pretrain_model_path "./models/ClickPose_model_only_R50.pth" \
    --eval
```
</details>


<details>
  <summary>Neural Interactive-NoC metric</summary>

```
export CLICKPOSE_COCO_PATH=/path/to/your/coco_dir
export CLICKPOSE_NoC_Test="TRUE"
    python -m torch.distributed.launch --nproc_per_node=4 --master_port 3458 main.py \
    --output_dir "logs/ClickPose_Neural_Interactive_eval" \
    -c config/clickpose.cfg.py \
    --options batch_size=4 epochs=100 lr_drop=80 use_ema=TRUE human_feedback=TRUE feedback_loop_NOC_test=TRUE feedback_inference=TRUE only_correction=FALSE num_select=20 
    --dataset_file="coco"
    --pretrain_model_path "./models/ClickPose_interactive_R50.pth" \
    --eval
```
</details>


<details>
  <summary>Neural Interactive-AP metric</summary>

```
export CLICKPOSE_COCO_PATH=/path/to/your/coco_dir
export CLICKPOSE_NoC_Test="TRUE"
for CLICKPOSE_Click_Number in {1..17}
do
    python -m torch.distributed.launch --nproc_per_node=4 --master_port 3458 main.py \
    --output_dir "logs/ClickPose_Neural_Interactive_eval" \
    -c config/clickpose.cfg.py \
    --options batch_size=4 epochs=100 lr_drop=80 use_ema=TRUE human_feedback=TRUE feedback_loop_NOC_test=FALSE feedback_inference=TRUE only_correction=FALSE num_select=20 
    --dataset_file="coco"
    --pretrain_model_path "./models/ClickPose_interactive_R50.pth" \
    --eval
done


```
</details>



### Evaluation on Human-Art:

<details>
  <summary>Model-Only</summary>

```
export CLICKPOSE_HumanArt_PATH=/path/to/your/humanart_dir
 python -m torch.distributed.launch --nproc_per_node=4 main.py \
    --output_dir "logs/ClickPose_Model-Only_eval" \
    -c config/clickpose.cfg.py \
    --options batch_size=4 epochs=100 lr_drop=80 use_ema=TRUE human_feedback=FLASE feedback_loop_NOC_test=FALSE feedback_inference=FALSE only_correction=FALSE
    --dataset_file="humanart"
    --pretrain_model_path "./models/ClickPose_model_only_R50.pth" \
    --eval
```
</details>


<details>
  <summary>Neural Interactive-NoC metric</summary>

```
export CLICKPOSE_HumanArt_PATH=/path/to/your/humanart_dir
export CLICKPOSE_NoC_Test="TRUE"
    python -m torch.distributed.launch --nproc_per_node=4 --master_port 3458 main.py \
    --output_dir "logs/ClickPose_Neural_Interactive_eval" \
    -c config/clickpose.cfg.py \
    --options batch_size=4 epochs=100 lr_drop=80 use_ema=TRUE human_feedback=TRUE feedback_loop_NOC_test=TRUE feedback_inference=TRUE only_correction=FALSE num_select=20 
    --dataset_file="humanart"
    --pretrain_model_path "./models/ClickPose_interactive_R50.pth" \
    --eval
```
</details>


<details>
  <summary>Neural Interactive-AP metric</summary>

```
export CLICKPOSE_HumanArt_PATH=/path/to/your/humanart_dir
export CLICKPOSE_NoC_Test="TRUE"
for CLICKPOSE_Click_Number in {1..17}
do
    python -m torch.distributed.launch --nproc_per_node=4 --master_port 3458 main.py \
    --output_dir "logs/ClickPose_Neural_Interactive_eval" \
    -c config/clickpose.cfg.py \
    --options batch_size=4 epochs=100 lr_drop=80 use_ema=TRUE human_feedback=TRUE feedback_loop_NOC_test=FALSE feedback_inference=TRUE only_correction=FALSE num_select=20 
    --dataset_file="humanart"
    --pretrain_model_path "./models/ClickPose_interactive_R50.pth" \
    --eval
done


```
</details>



### Evaluation on OCHuman:

<details>
  <summary>Model-Only</summary>

```
export CLICKPOSE_OCHuman_PATH=/path/to/your/ochuman_dir
 python -m torch.distributed.launch --nproc_per_node=4 main.py \
    --output_dir "logs/ClickPose_Model-Only_eval" \
    -c config/clickpose.cfg.py \
    --options batch_size=4 epochs=100 lr_drop=80 use_ema=TRUE human_feedback=FLASE feedback_loop_NOC_test=FALSE feedback_inference=FALSE only_correction=FALSE
    --dataset_file="ochuman"
    --pretrain_model_path "./models/ClickPose_model_only_R50.pth" \
    --eval
```
</details>


<details>
  <summary>Neural Interactive-NoC metric</summary>

```
export CLICKPOSE_OCHuman_PATH=/path/to/your/ochuman_dir
export CLICKPOSE_NoC_Test="TRUE"
    python -m torch.distributed.launch --nproc_per_node=4 --master_port 3458 main.py \
    --output_dir "logs/ClickPose_Neural_Interactive_eval" \
    -c config/clickpose.cfg.py \
    --options batch_size=4 epochs=100 lr_drop=80 use_ema=TRUE human_feedback=TRUE feedback_loop_NOC_test=TRUE feedback_inference=TRUE only_correction=FALSE num_select=20 
    --dataset_file="ochuman"
    --pretrain_model_path "./models/ClickPose_interactive_R50.pth" \
    --eval
```
</details>


<details>
  <summary>Neural Interactive-AP metric</summary>

```
export CLICKPOSE_OCHuman_PATH=/path/to/your/ochuman_dir
export CLICKPOSE_NoC_Test="TRUE"
for CLICKPOSE_Click_Number in {1..17}
do
    python -m torch.distributed.launch --nproc_per_node=4 --master_port 3458 main.py \
    --output_dir "logs/ClickPose_Neural_Interactive_eval" \
    -c config/clickpose.cfg.py \
    --options batch_size=4 epochs=100 lr_drop=80 use_ema=TRUE human_feedback=TRUE feedback_loop_NOC_test=FALSE feedback_inference=TRUE only_correction=FALSE num_select=20 
    --dataset_file="ochuman"
    --pretrain_model_path "./models/ClickPose_interactive_R50.pth" \
    --eval
done


```
</details>




### Cite Click-Pose

```angular2html

```


