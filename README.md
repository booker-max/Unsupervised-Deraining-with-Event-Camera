# 🚀Unsupervised-Deraining-with-Event-Camera (ICCV2023)

## The is official PyTorch implementation of paper "Unsupervised Video Deraining with An Event Camera".

Jin Wang, Yueyi Zhang*, Wenming Weng, Zhiwei Xiong

University of Science and Technology of China (USTC), Hefei, China

Institute of Artificial Intelligence, Hefei Comprehensive National Science Center, Hefei, China

*Corresponding Author

<div align=center>
<img src="images/contrastive.PNG">
</div>

I really like the work [Contrastive Multiview Coding](https://arxiv.org/pdf/1906.05849.pdf) and was greatly inspired.
🚀
**"Humans view the world through many sensory channels, e.g., the long-wavelength light channel, viewed by the left eye, or the high-frequency vibrations channel, heard by the right ear. Each view is noisy and incomplete, but important factors, such as physics, geometry, and semantics, tend to be shared between all views (e.g., a “dog” can be seen, heard, and felt)."**




## Abstract
Current unsupervised video deraining methods are inefficient in modeling the intricate spatio-temporal properties of rain, which leads to unsatisfactory results. In this paper, we propose a novel approach by integrating a bio-inspired event camera into the unsupervised video deraining pipeline, which enables us to capture high temporal resolution information and model complex rain characteristics. Specifically, we first design an end-to-end learning-based network consisting of two modules, the asymmetric separation module and the cross-modal fusion module. The two modules are responsible for segregating the features of the rain-background layer, and for positive enhancement and negative suppression from a cross-modal perspective, respectively.
Second, to regularize the network training, we elaborately design a cross-modal contrastive learning method that leverages the complementary information from event cameras, exploring the mutual exclusion and similarity of rain-background layers in different domains. This encourages the deraining network to focus on the distinctive characteristics of each layer and learn a more discriminative representation.
Moreover, we construct the first real-world dataset comprising rainy videos and events using a hybrid imaging system. Extensive experiments demonstrate the superior performance of our method on both synthetic and real-world datasets.

## Enviromenent
The entire network is implemented using PyTorch 1.6, Python 3.8, CUDA 11.3 on two NVIDIA GTX1080Ti GPUs.
```shell
sh requirements.sh
```

## Training stage
   ```
   CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node = 1 --master_port 29501 main.py
   ```

## Contact

If you have any problem with the released code and dataset, please contact me by email (jin01wang@mail.ustc.edu.cn).
