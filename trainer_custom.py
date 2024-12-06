# %%
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
import warnings
import sys
import io
import torch.nn.utils.prune as prune
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from demucs.transformer import MyTransformerEncoderLayer, CrossTransformerEncoderLayer, dynamic_sparse_attention, MultiheadAttention, scaled_dot_product_attention
from torch.quantization import quantize_dynamic
from fractions import Fraction
import kd_helper
from demucs.solver import Solver
import logging
from demucs import distrib
import hydra
from hydra.core.global_hydra import GlobalHydra
from dora import hydra_main
logger = logging.getLogger(__name__)

# %%
from demucs.separate import Separator

device = "cuda" if torch.cuda.is_available() else "cpu"
separator = Separator(
    model="htdemucs",
    repo=None,
    device=device,
    shifts=1,
    overlap=0.25,
    split=True,
    segment=None,
    jobs=None,
    callback=print
)
segment = None
callback = None
length = None
samplerate = 44100
device

# %%
# Function to count the number of parameters of a torch model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# %%
student_model, teacher_model = kd_helper.get_student_teacher_models(partial_weight_copy=True)

# %%
print(f"{count_parameters(teacher_model):,} parameters in teacher model")
print(f"{count_parameters(student_model):,} parameters in student model")

# %%
audio_input = torch.randn(1, 2, 44100*5)  # Example input
# Forward pass through the model
teacher_start = time.time()
with torch.no_grad():
    teacher_separated_sources = teacher_model(audio_input)
teacher_end = time.time()
print("Time taken for teacher model: ", teacher_end - teacher_start)
print("Teacher model output shape: ", teacher_separated_sources.shape)
student_start = time.time()
with torch.no_grad():
    # student_separated_sources = student_model(audio_input[:, :, ::2])
    student_separated_sources = student_model(audio_input)
student_end = time.time()
print("Time taken for student model: ", student_end - student_start)
print("Student model output shape: ", student_separated_sources.shape)

# %%
class Args:
    def __init__(self):
        self.seed = 42
        self.batch_size = 8
        self.epochs = 50
        self.downsample_factor = 2
        # Dataset related arguments
        self.dset = self.DatasetArgs()
        # Optimization related arguments
        self.optim = self.OptimArgs()
        # Augmentation related arguments
        self.augment = self.AugmentArgs()
        # Testing related arguments
        self.test = self.Test()
        # Miscellaneous arguments
        self.misc = self.MiscArgs()
        self.sources = ['drums', 'bass', 'other', 'vocals']
        self.sequence_length_in_seconds = 5
        self.save_folder = "MyTrainingOutputV3/"

    class DatasetArgs:
        def __init__(self):
            self.musdb = r'/home/ubuntu/distillation_demucs/musdb18hq'
            self.musdb_samplerate = 44100
            self.use_musdb = True
            self.wav = None  # path to custom wav dataset
            self.wav2 = None  # second custom wav dataset
            self.segment = 11
            self.shift = 1
            self.train_valid = False
            self.full_cv = True
            self.samplerate = 44100
            self.channels = 2
            self.normalize = True
            self.metadata = './metadata'
            self.sources = ['drums', 'bass', 'other', 'vocals']
            self.valid_samples = None  # valid dataset size
            self.backend = None

    class OptimArgs:
        def __init__(self):
            self.lr = 3e-4
            self.momentum = 0.9
            self.beta2 = 0.999
            self.loss = 'l1'  # l1 or mse
            self.optim = 'adam'
            self.weight_decay = 0
            self.clip_grad = 0

    class AugmentArgs:
        def __init__(self):
            self.shift_same = False
            self.repitch = self.Repitch()
            self.remix = self.Remix()
            self.scale = self.Scale()
            self.flip = True

        class Repitch:
            def __init__(self):
                self.proba = 0.2
                self.max_tempo = 12

        class Remix:
            def __init__(self):
                self.proba = 1
                self.group_size = 4
        
        class Scale:
            def __init__(self):
                self.proba = 1
                self.min = 0.25
                self.max = 1.25

    class Test:
        def __init__(self):
            self.save = False
            self.best = True
            self.workers = 2
            self.every = 5
            self.split = True
            self.shifts = 1
            self.overlap = 0.25
            self.sdr = True
            self.metric = 'loss'
            self.nonhq = None

    class MiscArgs:
        def __init__(self):
            # You can add any other miscellaneous arguments here if needed.
            self.show = False
            self.num_workers = 6
            self.num_prints = 4
            self.verbose = False

# Initialize args object with default values from the config file
args = Args()

# Accessing a parameter would be like this:
print(args.dset.musdb)

# %%
import demucs.train
from kt_solver import KTSolver
from demucs.repitch import RepitchedWrapper


# train_set, valid_set = demucs.train.get_datasets(args)
device = "cuda" if torch.cuda.is_available() else "cpu"
distrib.init()

def get_my_solver(args, model_only=False):
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
            vocals.append(args.sources.index('vocals'))
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

# %%
train_set, valid_set = demucs.train.get_datasets(args)
train_loader = distrib.loader(
    train_set, batch_size=args.batch_size, shuffle=True,
    num_workers=args.misc.num_workers, drop_last=True)
valid_loader = distrib.loader(
    valid_set, batch_size=1, shuffle=False,
    num_workers=args.misc.num_workers)

# %%
def apply_downsampling(wav_batch, downsample_factor):
    assert len(wav_batch.shape) == 4, f"Expected 4D tensor, got {len(wav_batch.shape)}D tensor"
    return wav_batch[:, :, :, ::downsample_factor]

def get_random_slice(wav_batch, segment_length_in_seconds, seed=None):
    bs, num_sources, channels, num_timesteps = wav_batch.shape
    total_length = args.dset.samplerate * segment_length_in_seconds
    if seed is not None:
        np.random.seed(seed)
    if num_timesteps <= total_length:
        return wav_batch
    random_start = np.random.randint(0, num_timesteps - total_length)
    ret = wav_batch[:, :, :, random_start:random_start + total_length]
    assert ret.shape == (bs, num_sources, channels, total_length), f"Expected shape {(bs, num_sources, channels, total_length)}, got {ret.shape}"
    return ret

# %%
splits_names = args.sources
def calculate_sdr(target: torch.Tensor, estimate: torch.Tensor) -> float:
    """
    Calculate the Signal-to-Distortion Ratio (SDR).

    Args:
        target (torch.Tensor): The ground truth signal of shape (2, timesteps).
        estimate (torch.Tensor): The estimated signal of shape (2, timesteps).

    Returns:
        float: The SDR value in decibels.
    """
    # Ensure the input tensors are of the same shape
    assert target.shape == estimate.shape, "Target and estimate must have the same shape."
    target_norm_squared = torch.norm(target, p=2) ** 2

    error = target - estimate
    error_norm_squared = torch.norm(error, p=2) ** 2

    sdr = 10 * torch.log10(target_norm_squared / error_norm_squared)
    return sdr.item()

def calculate_sdr_stem(target_list, model_pred_list):
    sdr_list = []
    for target, model_pred in zip(target_list, model_pred_list):
        sdr_list.append(calculate_sdr(target, model_pred))
    return np.array(sdr_list).mean()

def get_sdr_all(actual_audio, model_audio):
    sdr_list = []
    for split in splits_names:
        sdr_list.append(calculate_sdr_stem(actual_audio[split], model_audio[split]))
    return np.array(sdr_list).mean()

# %%
def get_audios_from_batch(sources_batch, estimates_batch):
    actual_audio = {i: [] for i in splits_names}
    model_audio = {i: [] for i in splits_names}
    assert sources_batch.shape == estimates_batch.shape, f"Expected {sources_batch.shape} == {estimates_batch.shape}"
    bs, num_sources, num_channels, num_samples = sources_batch.shape
    assert num_sources == 4, f"Expected 4 sources, got {num_sources}"
    for ind, split in enumerate(splits_names):
        for i in range(bs):
            actual_audio[split].append(sources_batch[i, ind])
            model_audio[split].append(estimates_batch[i, ind])
    return actual_audio, model_audio

# %%
from demucs.apply import apply_model
def evaluate(model, loader, is_student_model=False):
    model.to(device)
    with torch.no_grad():
        model.eval()
        total_actual_audio = {i: [] for i in splits_names}
        total_model_audio = {i: [] for i in splits_names}
        
        for ind, sources in enumerate(tqdm(loader)):
            sources = sources.to(device)
            bs, num_sources, channels, num_timesteps = sources.shape
            sources = get_random_slice(sources, args.sequence_length_in_seconds, seed=ind)
            if is_student_model:
                sources = apply_downsampling(sources, args.downsample_factor)
            if num_sources == 5:
                mix = sources[:, 0]
                sources = sources[:, 1:]
                estimates = apply_model(model, mix, split=args.test.split, overlap=0)
                actual_audio, model_audio = get_audios_from_batch(sources, estimates)
            elif num_sources == 4:
                mix = sources.sum(dim=1)
                estimates = apply_model(model, mix, split=args.test.split, overlap=0)
                actual_audio, model_audio = get_audios_from_batch(sources, estimates)
            else:
                assert False, f"Expected 4 or 5 sources, got {num_sources}"
            
            for split in splits_names:
                total_actual_audio[split].extend(actual_audio[split])
                total_model_audio[split].extend(model_audio[split])
        sdr = get_sdr_all(total_actual_audio, total_model_audio)
        return sdr
# evaluate(teacher_model, valid_loader, is_student_model=False)
# evaluate(student_model, valid_loader, is_student_model=True)

# %%
def get_my_optimizer(model):
    my_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.optim.lr,
            betas=(args.optim.momentum, args.optim.beta2),
            weight_decay=args.optim.weight_decay,
        )
    return my_optimizer

# %%
import os
import torch
import torch.nn as nn
from tqdm import tqdm

def save_model(student_model, args, epoch_num):
    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)
    torch.save(student_model.state_dict(), os.path.join(save_folder, f"student_model_epoch{epoch_num}.pth"))
    print(f"Saved student model to {save_folder}")

def kd_train(teacher_model, student_model, train_loader, valid_loader, args, debug=False):
    student_model.train()
    teacher_model.eval()
    student_model.to(device)
    teacher_model.to(device)
    if debug:
        print("Student model and teacher model moved to device")
    optimizer = get_my_optimizer(student_model)
    criterion = nn.MSELoss()
    
    if debug:
        print("Starting training loop")
    for epoch in range(args.epochs):
        running_loss = 0.0
        num_batches = 0
        student_model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")

        for ind, sources in enumerate(progress_bar):
            if debug:
                print(f"Starting batch {ind+1}")
            sources = sources.to(device)
            bs, num_sources, channels, num_timesteps = sources.shape
            sources = get_random_slice(sources, args.sequence_length_in_seconds)
            og_mix = sources.sum(dim=1)

            with torch.no_grad():
                teacher_estimates = apply_model(teacher_model, og_mix, split=args.test.split, overlap=0)
                assert teacher_estimates.shape == sources.shape, f"Expected {teacher_estimates.shape} == {sources.shape}"
            if debug:
                print(f"Teacher estimates shape: {teacher_estimates.shape}")

            sources = apply_downsampling(sources, args.downsample_factor)
            mix = sources.sum(dim=1)
            teacher_estimates = apply_downsampling(teacher_estimates, args.downsample_factor)

            # Forward pass for student model
            if debug:
                print(f"Mix shape: {mix.shape}")
                print("Starting forward pass for student model")
            student_estimates = student_model(mix)
            if debug:
                print(f"Student estimates shape: {student_estimates.shape}")
            
            assert student_estimates.shape == sources.shape, f"Expected {student_estimates.shape} == {sources.shape}"

            # Compute loss
            if debug:
                print("Computing loss")
            loss = criterion(student_estimates, teacher_estimates)
            if debug:
                print(f"Loss: {loss.item()}")

            # Backpropagation and optimization step
            if debug:
                print("Backpropagation and optimization step")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if debug:
                print("Backpropagation and optimization step done")

            # Update running loss
            running_loss += loss.item()
            num_batches += 1

            # Update progress bar with running average loss
            avg_loss = running_loss / num_batches
            progress_bar.set_postfix(loss=avg_loss)
            if debug:
                print(f"Batch {ind+1} completed")
                return

        # Evaluate on validation set and report SDR
        valid_sdr = evaluate(student_model, valid_loader, is_student_model=True)
        print(f"Validation SDR: {valid_sdr:.2f}")

        # Save model at the end of each epoch
        save_model(student_model, args, epoch + 1)

# %%
checkpoint_path = "MyTrainingOutputV2/student_model_epoch40.pth"
if checkpoint_path is not None:
    student_model.load_state_dict(torch.load(checkpoint_path))
    print(f"Loaded student model from {checkpoint_path}")

# %%
initial_sdr = evaluate(student_model, valid_loader, is_student_model=True)
print(f"Initial student model SDR: {initial_sdr:.2f}")

# %%
kd_train(teacher_model, student_model, train_loader, valid_loader, args, debug=False)

# %%
save_model(student_model, args, 0.5)

# %%



