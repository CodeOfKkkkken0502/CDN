## 硕士学位论文代码

2131488 庄子鲲

## 环境搭建
安装所需包
```
pip install -r requirements.txt
```
安装CLIP
```
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

## 数据集下载

### HICO-DET
[下载链接](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk)
。下载后将`hico_20160224_det.tar.gz`解压至`data`目录。

另外需要下载PPDM作者的注释文件：[下载链接](https://drive.google.com/open?id=1WI-gsNLS-t0Kh8TVki1wXqc3y2Ow1f2R)，将注释文件如下所示放置在`data`目录中：
```
data
 └─ hico_20160224_det
     |─ annotations
     |   |─ trainval_hico.json
     |   |─ test_hico.json
     |   └─ corre_hico.npy
     :
```

### V-COCO
首先克隆[V-COCO仓库](https://github.com/s-gupta/v-coco)，按照仓库中的指示编译生成`instances_vcoco_all_2014.json`文件，然后下载`prior.pickle`文件[下载链接](https://drive.google.com/drive/folders/10uuzvMUCVVv95-xAZg5KS94QXm7QXZW4)，
将文件如下所示放置在`data`目录中：
```
CDN
 |─ data
 │   └─ v-coco
 |       |─ data
 |       |   |─ instances_vcoco_all_2014.json
 |       |   :
 |       |─ prior.pickle
 |       |─ images
 |       |   |─ train2014
 |       |   |   |─ COCO_train2014_000000000009.jpg
 |       |   |   :
 |       |   └─ val2014
 |       |       |─ COCO_val2014_000000000042.jpg
 |       |       :
 |       |─ annotations
 :       :
```
运行以下命令，将PPDM注释文件转换为HOI-A格式：
```
PYTHONPATH=data/v-coco \
        python convert_vcoco_annotations.py \
        --load_path data/v-coco/data \
        --prior_path data/v-coco/prior.pickle \
        --save_path data/v-coco/annotations
```
执行此命令时需使用Python2，Python3会在`vsrl_utils.py`中报错。成功执行后，在`annotations`目录会生成HOI-A格式的注释文件`corre_vcoco.npy`, `test_vcoco.json`，`trainval_vcoco.json`。



## 预训练DETR参数
下载DETR预训练参数[下载链接](https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth)，将其放置于`params`目录。
```
python convert_parameters.py \
        --load_path params/detr-r50-e632da11.pth \
        --save_path params/detr-r50-pre-2stage-q64.pth \
        --num_queries 64

python convert_parameters.py \
        --load_path params/detr-r50-e632da11.pth \
        --save_path params/detr-r50-pre-2stage.pth \
        --dataset vcoco
```

## 训练
训练分为两步，首先训练整个模型，之后对解码器和前馈网络进行微调训练。
### HICO-DET
```
torchrun --nproc_per_node=2 main.py \
        --pretrained params/detr-r50-pre-2stage-q64.pth \
        --output_dir logs \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers_hopd 6 \
        --dec_layers_interaction 6 \
        --epochs 90 \
        --lr_drop 60 \
        --use_nms_filter \
        --compo \  #第3章方法
        --compo_new \ #第3章方法
        --batch_weight_mode 1 \ #第3章方法
        --uncertainty \ #第4章方法
        --superclass \ #第5章方法
        --clip_visual \ #第5章方法

torchrun --nproc_per_node=2 main.py \
        --pretrained logs/checkpoint_last_90.pth \
        --output_dir logs \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers_hopd 6 \
        --dec_layers_interaction 6 \
        --epochs 10 \
        --freeze_mode 1 \
        --obj_reweight \
        --verb_reweight \
        --lr 1e-5 \
        --lr_backbone 1e-6 \
        --use_nms_filter \
        --compo \  #第3章方法
        --compo_new \ #第3章方法
        --batch_weight_mode 1 \ #第3章方法
        --uncertainty \ #第4章方法
        --superclass \ #第5章方法
        --clip_visual \ #第5章方法
```

### V-COCO
```
torchrun --nproc_per_node=2 main.py \
        --pretrained params/detr-r50-pre-2stage.pth \
        --output_dir logs \
        --dataset_file vcoco \
        --hoi_path data/v-coco \
        --num_obj_classes 81 \
        --num_verb_classes 29 \
        --backbone resnet50 \
        --num_queries 100 \
        --dec_layers_hopd 3 \
        --dec_layers_interaction 3 \
        --epochs 90 \
        --lr_drop 60 \
        --use_nms_filter \
        --compo \  #第3章方法
        --compo_new \ #第3章方法
        --batch_weight_mode 2 \ #第3章方法
        --uncertainty \ #第4章方法
        --superclass \ #第5章方法
        --clip_visual \ #第5章方法

torchrun --nproc_per_node=2 main.py \
        --pretrained logs/checkpoint_last.pth \
        --output_dir logs/ \
        --dataset_file vcoco \
        --hoi_path data/v-coco \
        --num_obj_classes 81 \
        --num_verb_classes 29 \
        --backbone resnet50 \
        --num_queries 100 \
        --dec_layers_hopd 3 \
        --dec_layers_interaction 3 \
        --epochs 10 \
        --freeze_mode 1 \
        --verb_reweight \
        --lr 1e-5 \
        --lr_backbone 1e-6 \
        --use_nms_filter \
        --compo \  #第3章方法
        --compo_new \ #第3章方法
        --batch_weight_mode 2 \ #第3章方法
        --uncertainty \ #第4章方法
        --superclass \ #第5章方法
        --clip_visual \ #第5章方法
```

## 性能评估

### HICO-DET

对于训练得到的模型参数`trained_params.pth`，运行以下命令进行性能评估：
```
torchrun --nproc_per_node=2 main.py \
        --pretrained trained_params.pth \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers_hopd 6 \
        --dec_layers_interaction 6 \
        --eval \
        --use_nms_filter \
        --compo \  #第3章方法
        --compo_new \ #第3章方法
        --batch_weight_mode 2 \ #第3章方法
        --uncertainty \ #第4章方法
        --superclass \ #第5章方法
        --clip_visual \ #第5章方法
```

### V-COCO
首先在`data/v-coco/vsrl_eval.py`的主函数中添加以下代码：
```
if __name__ == '__main__':
  import sys

  vsrl_annot_file = 'data/vcoco/vcoco_test.json'
  coco_file = 'data/instances_vcoco_all_2014.json'
  split_file = 'data/splits/vcoco_test.ids'

  vcocoeval = VCOCOeval(vsrl_annot_file, coco_file, split_file)

  det_file = sys.argv[1]
  vcocoeval._do_eval(det_file, ovr_thresh=0.5)
```

之后运行如下命令进行性能评估：
```
python generate_vcoco_official.py \
        --param_path pretrained/vcoco_cdn_s.pth \
        --save_path vcoco.pickle \
        --hoi_path data/v-coco \
        --dec_layers_hopd 3 \
        --dec_layers_interaction 3 \
        --use_nms_filter \
        --compo \  #第3章方法
        --compo_new \ #第3章方法
        --batch_weight_mode 2 \ #第3章方法
        --uncertainty \ #第4章方法
        --superclass \ #第5章方法
        --clip_visual \ #第5章方法

cd data/v-coco
python vsrl_eval.py vcoco.pickle
```


