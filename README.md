# Object Detection Ensemble

## Algorithm

Reference to the main idea: [https://github.com/ahrnbom/ensemble-objdet](https://github.com/ahrnbom/ensemble-objdet/blob/master/ensemble.py)  
The idea is used for ensembling the boxes that have same `image_id` and `category_id`.  

## Reproducing

The test has been conducted using COCO 2017 val dataset and [Detectron2](https://github.com/facebookresearch/detectron2).  
In order to reproduce what I've got during the test, you can follow the below.  

### 1. Prepare conda environment
1) Prepare Ubuntu 20.04 in WSL(Windows Subsystem for Linux) 2.  
~~The environment has been tested without using GPU due to WSL limitation.~~  
The environment has been tested using GPU thanks to 21H2 update for Windows 10 Pro.

2) Run the command line below to install the packages except for Detectron2.
``` shell
$ conda env create -f environment_wsl.yml
```

3) Activate the environment.
``` shell
$ conda activate torch
```

4) Install Detectron2.
``` shell
$ git clone https://github.com/facebookresearch/detectron2.git
$ cd detectron2 && git checkout v0.6 && cd ..
$ python -m pip install -e detectron2
```
When having any installation issue with Detectron2 please followed the instruction [here](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).


### 2. Prepare COCO 2017 val dataset
``` shell
$ wget http://images.cocodataset.org/zips/val2017.zip
$ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
$ mkdir -p datasets/coco
$ unzip val2017.zip -d datasets/coco
$ unzip annotations_trainval2017.zip -d datasets/coco 
```

### 3. Emit estimation results of coco_2017_val for object detection models
``` shell
$ python evaluate_pretrained_model.py
$ ls -A1 datasets/coco
coco_instances_results_faster_rcnn_R_50_DC5_1x.json
coco_instances_results_faster_rcnn_R_50_FPN_1x.json
coco_instances_results_faster_rcnn_R_50_FPN_3x.json
```

### 4. Create an ensemble and evaluate the output
``` shell
$ python evaluate_ensemble.py
Loaded output/coco_instances_results_faster_rcnn_R_50_DC5_1x.json
Loaded output/coco_instances_results_faster_rcnn_R_50_FPN_1x.json
Loaded output/coco_instances_results_faster_rcnn_R_50_FPN_3x.json
Saved the ensemble result to output/ensemble_results.json
Preparing for evaluating the ensemble
Loading and preparing results...
DONE (t=0.23s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
COCOeval_opt.evaluate() finished in 8.24 seconds.
Accumulating evaluation results...
COCOeval_opt.accumulate() finished in 0.87 seconds.
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.386
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.572
...
```

### 5. Try other models
Edit path for the models in `config.py` and repeat step 3 and 4.


| model                   | box AP (at IoU=0.50:0.95, area=all) | model                   | box AP (at IoU=0.50:0.95, area=all) | model                   | box AP (at IoU=0.50:0.95, area=all) |
|-------------------------|-------------------------------------|-------------------------|-------------------------------------|-------------------------|-------------------------------------|
| faster_rcnn_R_50_C4_1x  | 33.05873724953157                   | faster_rcnn_R_50_C4_1x  | 33.05873724953157                   | faster_rcnn_R_50_DC5_1x | 35.02630019513202                   |
| faster_rcnn_R_50_DC5_1x | 35.02630019513202                   | faster_rcnn_R_50_C4_3x  | 35.92233833824599                   | faster_rcnn_R_50_FPN_1x | 34.35279506894608                   |
| faster_rcnn_R_50_FPN_1x | 34.35279506894608                   | faster_rcnn_R_101_C4_3x | 38.51233125665915                   | faster_rcnn_R_50_FPN_3x | 36.66724655820816                   |
| Ensemble                | 36.988241172963086                  | Ensemble                | 39.57683208823928                   | Ensemble                | 38.61288520614146                   |

