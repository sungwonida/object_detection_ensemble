# AUTHOR: David Jung
# EMAIL: sungwonida@gmail.com
# DATE: 2020-09-06

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import libraries
import os, shutil
from pathlib import Path
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# paths for saving the results
from config import *

# define some helper functions
def evaluate_pretrained_model(config, dataset, save_to=None):
    cfg = get_cfg()
    # cfg.MODEL.DEVICE = "cpu"
    cfg.merge_from_file(model_zoo.get_config_file(config))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config)

    cfg.DATASETS.TRAIN = (dataset + "_train",)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    output_dir = "./output/"
    evaluator = COCOEvaluator(dataset + "_val", cfg, False, output_dir=output_dir)
    val_loader = build_detection_test_loader(cfg, dataset + "_val")
    print(inference_on_dataset(trainer.model, val_loader, evaluator))

    if save_to:
        dst = os.path.abspath(save_to)
        Path(os.path.dirname(dst)).mkdir(parents=True, exist_ok=True)
        shutil.move(os.path.join(output_dir, 'coco_instances_results.json'), dst)


if __name__ == '__main__':
    for config, path in zip(detectron_configs, model_eval_paths):
        evaluate_pretrained_model(config, "coco_2017", save_to=path)
