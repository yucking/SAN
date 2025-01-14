try:
    # ignore ShapelyDeprecationWarning from fvcore
    import warnings

    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass
import copy
import itertools
import logging
import os
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Set

import detectron2.utils.comm as comm
import torch

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.parallel")
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesSemSegEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from tabulate import tabulate

from san import (
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_san_config,
)
from san.data import build_detection_test_loader, build_detection_train_loader
from san.utils import WandbWriter, setup_wandb


class Trainer(DefaultTrainer):
    '''
    重写父类(DefaultTrainer)的 build_writers 方法，将默认的写入器(如Tensorboard)替换为自定义的 WandbWriter,用于记录训练过程中的信息到 Weights & Biases(wandb)
    '''
    def build_writers(self):
        writers = super().build_writers()
        # use wandb writer instead.
        writers[-1] = WandbWriter()
        return writers

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        '''
        功能:根据给定的数据集名称和配置构建评估器,支持不同类型的评估器,如语义分割评估器、Cityscapes评估器等。

        如果没有提供 output_folder,则使用默认路径 cfg.OUTPUT_DIR/inference。
        获取数据集的评估器类型 (evaluator_type)。
        如果是语义分割或 ADE20K 类型的数据集，则添加 SemSegEvaluator 到评估器列表。
        如果是 cityscapes_sem_seg 类型的数据集，使用 CityscapesSemSegEvaluator,并检查是否支持多机器。
        如果没有匹配的评估器类型，抛出 NotImplementedError。
        如果有多个评估器，返回 DatasetEvaluators 对象，包含所有评估器。
        如果只有一个评估器，直接返回该评估器。
        '''
        if output_folder is None: 
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )

        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        '''
        功能：根据配置构建训练数据加载器，并且使用自定义的 MaskFormerSemanticDatasetMapper 进行数据映射。

        '''
        # resue maskformer dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None # build the default mapper
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        '''
        功能：构建测试数据加载器。'''
        # Add dataset meta info.
        return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        '''
        功能：构建学习率调度器，使用配置文件中指定的学习率调度策略
        '''
        # use poly scheduler
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        '''
        功能：根据配置文件中的超参数构建优化器，支持 SGD 和 ADAMW 优化器，并根据不同条件（如梯度裁剪、不同模块的权重衰减）调整优化器的超参数'''
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM  # ??? ----没找到参数默认值
        weight_decay_embed_group = cfg.SOLVER.WEIGHT_DECAY_EMBED_GROUP 
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        '''
        为模型中的不同参数设置不同的优化超参数,如学习率(r)和权重衰减(weight_decay)
        '''
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad: # 如果参数不需要梯度，则跳过
                    continue
                # Avoid duplicating parameters
                if value in memo: # 如果参数已经在 memo 中，则跳过
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults) # 复制默认的超参数
                hyperparams["param_name"] = ".".join([module_name, module_param_name])
                if "side_adapter_network" in module_name:
                    hyperparams["lr"] = (
                        hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                    )
                # scale clip lr
                if "clip" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.CLIP_MULTIPLIER
                if any([x in module_param_name for x in weight_decay_embed_group]):
                    hyperparams["weight_decay"] = weight_decay_embed
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE # 获取梯度裁剪的阈值
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(
                        *[x["params"] for x in self.param_groups]
                    )
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val) # 使用的是缩放裁剪
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        # display the lr and wd of each param group in a table
        optim_info = defaultdict(list)
        total_params_size = 0
        for group in optimizer.param_groups:
            optim_info["Param Name"].append(group["param_name"])
            optim_info["Param Shape"].append(
                "X".join([str(x) for x in list(group["params"][0].shape)])
            )
            total_params_size += group["params"][0].numel()
            optim_info["Lr"].append(group["lr"])
            optim_info["Wd"].append(group["weight_decay"])
        # Counting the number of parameters 打印表格
        optim_info["Param Name"].append("Total")
        optim_info["Param Shape"].append("{:.2f}M".format(total_params_size / 1e6))
        optim_info["Lr"].append("-")
        optim_info["Wd"].append("-")
        table = tabulate(
            list(zip(*optim_info.values())),
            headers=optim_info.keys(),
            tablefmt="grid",
            floatfmt=".2e",
            stralign="center",
            numalign="center",
        )
        logger = logging.getLogger("san")
        logger.info("Optimizer Info:\n{}\n".format(table))
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        # 测试时间增强是一种在测试阶段对输入数据进行数据增强的技术，以提高模型的泛化能力。
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_san_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    if not args.eval_only:
        setup_wandb(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="san")
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
