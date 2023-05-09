
# CenterDisks
Repository for the paper 
<br> by Katia Jodogne-del Litto<sup>1</sup>, Guillaume-Alexandre Bilodeau<sup>1</sup>.
<br>
<sup>1</sup> Polytechnique Montréal

## Abstract


## Example results
Ground-truth      |  Disks covering
:-------------------------:|:-------------------------:
![](imgs/bielefeld_000000_026550_full_v1_zoom.png)  |  ![](imgs/bielefeld_000000_026550_full_v2_zoom.png)

## Requirements:
- python 3.8
- pytorch (1.8.0, cuda 10.2)
- mmdetection (for DCN)
- various common packages (opencv, numpy...)

## Folder organization:
- experiments/: scripts for the experiments.
- Data path expected usually is /store/datasets/DATASET_NAME (changeable in code)
- src/lib/trains/gaussiandet.py is the training file
- src/lib/datasets/sample/gaussiandet.py is the sampling file
- src/lib/detectors/gaussiandet.py is the detector file

## Help
For general debugging and help to run the scripts: <br>
- This code is built upon: https://github.com/xingyizhou/CenterNet and https://github.com/hu64/CenterPoly
- The pre-trained weights are all available at this location

## Results:

| Datasets (test sets) |   AP  | AP50% | Runtime (s) | weights                                                                                                                                                 |
|:--------------------:|:-----:|-------|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| cityscapes           | 16.64 | 39.42 | 0.045       | [link](https://polymtlca0-my.sharepoint.com/:u:/g/personal/katia_jodogne--del-litto_polymtl_ca/EQIz3Vm96pFNqIZBF-BcK48BzcEXAIEK35cDupRb0uTfmw?e=3Zb7Sq) |
| KITTI                |  8.86 | 26.86 | 0.045       | [link](https://polymtlca0-my.sharepoint.com/:u:/g/personal/katia_jodogne--del-litto_polymtl_ca/ETprojfO4-JInpakjuxcDnIBJVTLh1Oz1Pcv4JtLQTZ5HQ?e=8JBlft) |
| IDD                  | 17.40 | 45.10 | 0.045       | [link](https://polymtlca0-my.sharepoint.com/:u:/g/personal/katia_jodogne--del-litto_polymtl_ca/EeLV5WjLXSxEqOD84_YCjmABELKItvE4uamHZOa7od3Bvw)          |

## Models
https://polymtlca0-my.sharepoint.com/:f:/g/personal/katia_jodogne--del-litto_polymtl_ca/EtvCinAQKlNJiekvaNMbAkMBwfzJd9yghBvUMIHZiL6uBw

## Acknowledgements
The code for this paper is mainly built upon [CenterPoly](https://github.com/hu64/CenterPoly) and [CenterNet](https://github.com/xingyizhou/CenterNet), we would therefore like to thank the authors for providing the source code of their paper. We also acknowledge the support of the Natural Sciences and Engineering Research Council of Canada (NSERC), and the support of IVADO [MSc-2022-4713306544].

## License
CenterPoly v2 is released under the MIT License. Portions of the code are borrowed from [CenterPoly](https://github.com/hu64/CenterPoly), [CenterNet](https://github.com/xingyizhou/CenterNet), [CornerNet](https://github.com/princeton-vl/CornerNet) (hourglassnet, loss functions), [dla](https://github.com/ucbdrive/dla) (DLA network), [mmdetection](https://github.com/open-mmlab/mmdetection)(deformable convolutions), and [cityscapesScripts](https://github.com/mcordts/cityscapesScripts) (cityscapes dataset evaluation). Please refer to the original License of these projects (See [NOTICE](NOTICE)).
