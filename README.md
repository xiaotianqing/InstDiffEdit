# InstDiffEdit
This repository contains the implementation of the AAAI 2024 paper:
> **Towards Efficient Diffusion-Based Image Editing with Instant Attention Masks** 
> [[Paper]](https://arxiv.org/abs/2401.07709) [[AAAI]](https://ojs.aaai.org/index.php/AAAI/article/view/28622)<br>
> Siyu Zou<sup>1</sup>, Jiji Tang<sup>2</sup>, Yiyi Zhou<sup>1</sup>, Jing He<sup>1</sup>, Chaoyi Zhao<sup>2</sup>, Rongsheng Zhang<sup>2</sup>, Zhipeng Hu<sup>2</sup>, [Xiaoshuai Sun](https://sites.google.com/view/xssun)<sup>1</sup><br>
><sup>1</sup>Key Laboratory of Multimedia Trusted Perception and Efficient Computing, Ministry of Education of China, Xiamen University<br>
> <sup>2</sup>Fuxi AI Lab, NetEase Inc., Hangzhou
 
 ## Model Architecture
![Model_architecture](https://github.com/xiaotianqing/InstDiffEdit/blob/main/figure/instdiffedit.jpg)


## Code Path

#### Code Structures
There are four parts in the code.
- **model**: It contains the implement files for InstDiffEdit, DiffEdit and SDEdit.
- **dataset_txt**: It contains the data splits of Imagen, ImageNet and Editing-Mask dataset.
- **dataset**: It contains the image and mask of Editing-Mask dataset.
- `.sh`: The inference scripts for InstDiffEdit.

## Dependencies

- ```Python 3.8```
- ```PyTorch == 1.13.1```
- ```Transformers == 4.25.1```
- ```diffusers == 0.8.0```
- ```NumPy```
- All experiments are performed with one A30 GPU.

## Datasets
There are two pdataset we used.
- **ImageNet**: We follow the evaluation protocol of FlexIT (https://github.com/facebookresearch/semanticimagetranslation). We obtained 1092 test images and made changes to the image category.
- **Imagen**: We use the 360 image with structured text prompts generated by Imagen(https://imagen.research.google/).
- **Editing-Mask**: 200 images show in dataset.

## Eval & Sample

Sample begin:
```shell
bash sample_begin.sh
```

Run in the Imagen or ImageNet or Editing-Mask:

```shell
bash run.sh
```

**Note**: 
- Diffedit and SDEdit can be used by the `.sh` file with some parameter changes.
- you can open the `.sh` file for parameter modification.
