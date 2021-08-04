# Anatomical-awareness-in-the-electric-field-estimation
## There is the support file for the conference paper: anatomical awareness improves the accuracy of the real-time electric field estimation

### Here only support for below models for the anatomcial segmentation or electric estimation:

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

Liang Ma, & Gangliang Zhong et al., (2021) anatomical awareness improves the accuracy of the real-time electric field estimation, processedings of IJCNN 2021, China ShenZhen.
