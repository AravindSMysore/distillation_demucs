from demucs import pretrained
import torch
from demucs.demucs import Demucs
from demucs.hdemucs import HDemucs
from demucs.apply import tensor_chunk
from demucs.htdemucs import HTDemucs
from demucs.utils import center_trim
from demucs.apply import TensorChunk
from demucs.audio import AudioFile, convert_audio, save_audio
from pathlib import Path
import demucs
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import scipy
from scipy.signal import resample, butter, filtfilt, cheby1
import os
import numpy as np
import torch
import sys
from fractions import Fraction
import kd_helper
from demucs.solver import Solver
import logging
from demucs import distrib
import hydra
from hydra.core.global_hydra import GlobalHydra
from dora import hydra_main
logger = logging.getLogger(__name__)
from demucs.separate import Separator
import demucs.train
from kt_solver import KTSolver
from demucs.repitch import RepitchedWrapper

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["HYDRA_FULL_ERROR"] = "1"


def get_my_solver(args, model_only=False):
    distrib.init()
    torch.manual_seed(args.seed)
    teacher_model, student_model = kd_helper.get_student_teacher_models(partial_weight_copy=True)
    if args.misc.show:
        mb = sum(p.numel() for p in teacher_model.parameters()) * 4 / 2**20
        print(f"Teacher model has {mb:.1f}MB")
        smb = sum(p.numel() for p in student_model.parameters()) * 4 / 2**20
        print(f"Student model has {smb:.1f}MB")
        if hasattr(teacher_model, "valid_length"):
            field = teacher_model.valid_length(1)
            print(f"Field: {field/args.dset.samplerate*1000:.1f}ms")
        sys.exit(0)

    teacher_model.to(device)
    student_model.to(device)
    
    optimizer = demucs.train.get_optimizer(student_model, args)
    
    assert args.batch_size % distrib.world_size == 0
    args.batch_size //= distrib.world_size
    
    if model_only:
        return KTSolver(None, student_model, teacher_model, optimizer, args)
    
    train_set, valid_set = demucs.train.get_datasets(args)
    
    if args.augment.repitch.proba:
        vocals = []
        if 'vocals' in args.dset.sources:
            vocals.append(args.dset.sources.index('vocals'))
        else:
            logger.warning("No vocal source found")
        if args.augment.repitch.proba:
            train_set = RepitchedWrapper(train_set, vocals=vocals, **args.augment.repitch)
    
    logger.info("train/valid set size: %d %d", len(train_set), len(valid_set))
    train_loader = distrib.loader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.misc.num_workers, drop_last=True)
    if args.dset.full_cv:
        valid_loader = distrib.loader(
            valid_set, batch_size=1, shuffle=False,
            num_workers=args.misc.num_workers)
    else:
        valid_loader = distrib.loader(
            valid_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.misc.num_workers, drop_last=True)
    loaders = {"train": train_loader, "valid": valid_loader}

    # Construct Solver
    return KTSolver(loaders, student_model, teacher_model, optimizer, args)

@hydra_main(config_path="./conf", config_name="config", version_base="1.1")
def main(args):
    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    for attr in ["musdb", "wav", "metadata"]:
        val = getattr(args.dset, attr)
        if val is not None:
            setattr(args.dset, attr, hydra.utils.to_absolute_path(val))

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    if args.misc.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)
    from dora import get_xp
    logger.debug(get_xp().cfg)

    solver = get_my_solver(args)
    solver.train()


if '_DORA_TEST_PATH' in os.environ:
    main.dora.dir = Path(os.environ['_DORA_TEST_PATH'])


if __name__ == "__main__":
    main()