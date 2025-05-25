# 目标检测与实例分割实验

本项目使用MMDetection框架在VOC数据集上训练和测试Mask R-CNN和Sparse R-CNN模型。

## 环境配置
1. 安装依赖：
```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab

# CUDA 11.1
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
conda install pytorch torchvision -c pytorch

pip install -U openmim
pip install --force-reinstall charset-normalizer==3.1.0
mim install "mmengine>=0.7.0"
mim install "mmcv>=2.0.0rc4"
```

2. 安装MMDetection：
```bash
cd mmdetection
pip install -v -e .
```

## 数据集准备

1. 下载VOC数据集：
```bash
python tools/misc/download_dataset.py --dataset-name voc2012
```

2. 解压数据集：
```bash
tar xvf data/coco/VOCtrainval_11-May-2012.tar -C data
```

3. 将数据集转换为COCO格式：
```bash
python tools/dataset_converters/pascal_voc.py data/VOCdevkit -o data/VOCdevkit/coco --out-format coco
```

## 训练模型

1. 训练Mask R-CNN：
下载预训练模型 https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth
```bash
bash tools/dist_train.sh configs/pascal_voc/mask-rcnn_r50_fpn_1x_voc0712.py 2
```

2. 训练Sparse R-CNN：
```bash
bash tools/dist_train.sh configs/pascal_voc/sparse-rcnn_r50_fpn_1x_voc0712.py 2
```

## 可视化

见visualize.ipynb

## 实验结果

实验结果将保存在以下目录：
- Mask R-CNN结果：`mmdetection/work_dirs/mask-rcnn_r50_fpn_1x_voc0712`
- Sparse R-CNN结果：`HW2/mmdetection/work_dirs/sparse-rcnn_r50_fpn_1x_voc0712`

## 模型权重

训练好的模型权重可以从以下链接下载：
- Mask R-CNN：[https://pan.baidu.com/s/1ztjXSb_a1GTFhfU8C2X4vw?pwd=ifb4]
- Sparse R-CNN：[https://pan.baidu.com/s/1b8J4e9ocPJT2tEdzjauu7A?pwd=2zef]
