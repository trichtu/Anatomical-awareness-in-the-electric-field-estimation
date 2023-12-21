### Anatomical-awareness-in-the-electric-field-estimation
There is the support file for the conference paper: anatomical awareness improves the accuracy of the real-time electric field estimation

### software:
python 3.8

pytorch >=1.8


### In this article, the region is limited in the motor cortex region. Here only support for trained models for the anatomcial segmentation or electric estimation :

#### anatomical segmentation model:
'U_Net_Seg'

#### electric estimation model:
'U_Net3D','U_Net3D_Att','U_Net3D_Att_Seg','two_step'

’two_step‘ processing of electric estimation (including anatomical segmentation + electric regression model) is only processed in test mode


### To achive a training process
python main.py --mode train --model U_Net3D

### To achive a validation process
python main.py --mode test --model U_Net3D
or
python main.py --mode test --model two_step

### if there is any help for your work and insights, it would be better to cite:

@INPROCEEDINGS{9533894,
  author={Ma, Liang and Zhong, Gangliang and Yang, Zhengyi and Fan, Linzhong and Jiang, Tianzi},
  booktitle={2021 International Joint Conference on Neural Networks (IJCNN)}, 
  title={Multi-scale anatomical awareness improves the accuracy of the real-time electric field estimation}, 
  year={2021},
  volume={},
  number={},
  pages={1-7},
  doi={10.1109/IJCNN52387.2021.9533894}}

