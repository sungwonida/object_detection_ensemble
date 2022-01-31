import numpy as np

############################################################
# Helper function for the weighting
############################################################
softmax = lambda X: [np.e**x / sum(np.e**np.array(X)) for x in X]

############################################################
# Paths for Detecton2 configuration and the evaluation result
############################################################

# # TEST 1
# detectron_configs = ["COCO-Detection/faster_rcnn_R_50_C4_1x.yaml",
#                      "COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml",
#                      "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml",]
# model_eval_paths = ["output/coco_instances_results_faster_rcnn_R_50_C4_1x.json",
#                     "output/coco_instances_results_faster_rcnn_R_50_DC5_1x.json",
#                     "output/coco_instances_results_faster_rcnn_R_50_FPN_1x.json"]
# weights = [33.058, 35.025, 34.352]

# # TEST 2
# detectron_configs = ["COCO-Detection/faster_rcnn_R_50_C4_1x.yaml",
#                      "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml",
#                      "COCO-Detection/faster_rcnn_R_101_C4_3x.yaml",]
# model_eval_paths = ["output/coco_instances_results_faster_rcnn_R_50_C4_1x.json",
#                     "output/coco_instances_results_faster_rcnn_R_50_C4_3x.json",
#                     "output/coco_instances_results_faster_rcnn_R_101_C4_3x.json"]
# weights = [(33.058*1)**2, (35.922*2)**2, (38.512*3)**2]

# TEST 3
detectron_configs = ["COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml",
                     "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml",
                     "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",]
model_eval_paths = ['output/coco_instances_results_faster_rcnn_R_50_DC5_1x.json',
                    'output/coco_instances_results_faster_rcnn_R_50_FPN_1x.json',
                    'output/coco_instances_results_faster_rcnn_R_50_FPN_3x.json', ]
weights = softmax([35.026, 34.353, 36.667])
