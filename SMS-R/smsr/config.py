#!/usr/bin/env python3
"""Hyperparameters for SMS-R sparse path-phase localization."""

from __future__ import annotations
from typing import Optional, Tuple
import torch

class Config:
    TRAIN_PATH = 'Trained_Data'
    VAL_PATH = 'Validation_Data'
    TEST_PATH = 'Test_Data'
    ORDINATE_CSV = 'ordinate.csv'
    VALID_RP_CLASSES_JSON = 'valid_rp_classes_DRL.json'
    OUTPUT_MODEL_PTH = 'best_model_DRL.pth'
    OUTPUT_MEAN_NPY = 'mean_DRL.npy'
    OUTPUT_STD_NPY = 'std_DRL.npy'
    OUTPUT_TRAINING_CURVES_PNG = 'training_curves_DRL.png'
    OUTPUT_VAL_CONFUSION_PNG = 'val_confusion_matrix_DRL.png'
    OUTPUT_VAL_PER_CLASS_PNG = 'val_per_class_accuracy_support_DRL.png'
    OUTPUT_VAL_ERROR_SCATTER_PNG = 'val_error_scatter_DRL.png'
    OUTPUT_ERROR_CDF_PNG = 'error_cdf_global_DRL.png'
    OUTPUT_ERROR_CDF_BY_PATH_DEPTH_PNG = 'error_cdf_by_path_depth_DRL.png'
    OUTPUT_ERROR_CDF_TRAIN_TEST_PNG = 'error_cdf_k{k}_DRL_train_test.png'
    OUTPUT_TEST_BAR_PNG = 'test_localization_error_bar_DRL_train_test.png'
    OUTPUT_PICTURE_DIR = 'Test_picture_DRL'
    MAX_SEQ_LEN = 152
    SEG_LEN = 38
    NUM_PHASES = 6
    NUM_PATHS = 4
    PATH_DEPTHS: Tuple[int, ...] = (1, 2, 3, 4)
    CACHE_DRL_TAG = 'DRL_phgrid'
    CACHE_DRL_VERSION = 'v1'
    NUM_CLASSES = 52
    BATCH_SIZE = 4096
    EPOCHS = 50
    LR = 0.004
    K_VALUE = 3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_DATA_PARALLEL = True
    USE_AMP = True
    USE_TORCH_COMPILE = False
    LABEL_SMOOTHING = 0.0
    USE_WEIGHTED_SAMPLER = True
    USE_CLASS_WEIGHTS = True
    LOSS_XY_WEIGHT = 1.8
    EVAL_FUSE_XY_WEIGHT = 0.72
    EVAL_SOFTMAX_DENOM_SCALE = 1.2
    EVAL_TOPK_WEIGHT_MODE = 'full_softmax_no_renorm'
    EVAL_TOPK_PROB_MASS_DIV = 1.0
    RP_SPACING_M = 1.6
    EVAL_ERROR_OUTLIER_MAX_M = 50.0 * RP_SPACING_M
    RP_LIST = ['1-001', '1-002', '1-004', '2-001', '2-003', '3-002', '3-004', '4-001', '4-003', '5-002', '5-004', '6-001', '6-003', '7-001', '7-003', '8-001', '8-005', '8-008', '9-001', '9-003', '9-006', '9-008', '10-001', '10-003', '10-006', '10-008', '11-001', '11-003', '11-006', '11-008', '12-001', '12-003', '12-006', '12-008', '13-001', '13-003', '13-006', '13-008', '14-002', '14-004', '14-006', '14-008', '15-001', '15-003', '15-005', '15-006', '15-008', '16-001', '16-004', '16-005', '16-007', '16-008']
    DATALOADER_NUM_WORKERS = 16
    TORCH_NUM_INTRAOP_THREADS = 16
    ZSCORE_INPLACE_CHUNK_ROWS = 4096
    TRAIN_SCAN_READ_WORKERS = 0
    TRAIN_SCAN_USE_THREADS = True
    TEST_SCAN_READ_WORKERS = 64
    TEST_SCAN_BATCH_SIZE = 50000
    TEST_EVAL_BATCH_SIZE: Optional[int] = 4096
    TEST_EVAL_SINGLE_GPU = True
    TEST_DATALOADER_NUM_WORKERS: Optional[int] = 0
    TEST_DATALOADER_PREFETCH_FACTOR = 2
assert len(Config.RP_LIST) == Config.NUM_CLASSES
