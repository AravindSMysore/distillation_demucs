from demucs import pretrained
import torch
from demucs.htdemucs import HTDemucs
from fractions import Fraction


def transfer_weights(teacher_model, student_model):
    # Iterate over named parameters in both models
    for (teacher_name, teacher_param), (student_name, student_param) in zip(teacher_model.named_parameters(), student_model.named_parameters()):
        if 'weight' in teacher_name and 'weight' in student_name:
            # Check if dimensions are compatible
            if teacher_param.shape == student_param.shape:
                # Directly copy if shapes match
                student_param.data.copy_(teacher_param.data)
            else:
                # Handle linear layers
                if len(teacher_param.shape) == 2 and len(student_param.shape) == 2:
                    # Copy a subset of rows and columns
                    rows = min(teacher_param.shape[0], student_param.shape[0])
                    cols = min(teacher_param.shape[1], student_param.shape[1])
                    student_param.data[:rows, :cols].copy_(teacher_param.data[:rows, :cols])
                
                # Handle convolutional layers (Conv1d or Conv2d)
                elif len(teacher_param.shape) == 4 or len(teacher_param.shape) == 3:
                    # Copy a subset of channels
                    out_channels = min(teacher_param.shape[0], student_param.shape[0])
                    in_channels = min(teacher_param.shape[1], student_param.shape[1])
                    kernel_size = teacher_param.shape[-1]  # Assuming same kernel size
                    if len(teacher_param.shape) == 4:  # Conv2d
                        kernel_height = teacher_param.shape[2]
                        student_param.data[:out_channels, :in_channels, :kernel_height, :kernel_size].copy_(
                            teacher_param.data[:out_channels, :in_channels, :kernel_height, :kernel_size]
                        )
                    else:  # Conv1d
                        student_param.data[:out_channels, :in_channels, :kernel_size].copy_(
                            teacher_param.data[:out_channels, :in_channels, :kernel_size]
                        )
                
                # Handle deconvolutional layers (ConvTranspose1d or ConvTranspose2d)
                elif 'conv_tr' in teacher_name or 'conv_tr' in student_name:
                    out_channels = min(teacher_param.shape[0], student_param.shape[0])
                    in_channels = min(teacher_param.shape[1], student_param.shape[1])
                    kernel_size = teacher_param.shape[-1]  # Assuming same kernel size
                    if len(teacher_param.shape) == 4:  # ConvTranspose2d
                        kernel_height = teacher_param.shape[2]
                        student_param.data[:out_channels, :in_channels, :kernel_height, :kernel_size].copy_(
                            teacher_param.data[:out_channels, :in_channels, :kernel_height, :kernel_size]
                        )
                    else:  # ConvTranspose1d
                        student_param.data[:out_channels, :in_channels, :kernel_size].copy_(
                            teacher_param.data[:out_channels, :in_channels, :kernel_size]
                        )
    print("Partial weights transferred successfully from the teacher to the student model.")

def get_teacher_model():
    model_htdemucs = pretrained.get_model('htdemucs')
    # model_htdemucs.use_train_segment = False
    teacher_model = model_htdemucs.models[0]
    # teacher_model.use_train_segment = False
    return teacher_model

def get_student_model():
    teacher_kwargs = {'sources': ['drums', 'bass', 'other', 'vocals'], 'audio_channels': 2, 'samplerate': 44100, 'segment': Fraction(39, 5), 'channels': 48, 'channels_time': None, 'growth': 2, 'nfft': 4096, 'wiener_iters': 0, 'end_iters': 0, 'wiener_residual': False, 'cac': True, 'depth': 4, 'rewrite': True, 'multi_freqs': [], 'multi_freqs_depth': 3, 'freq_emb': 0.2, 'emb_scale': 10, 'emb_smooth': True, 'kernel_size': 8, 'stride': 4, 'time_stride': 2, 'context': 1, 'context_enc': 0, 'norm_starts': 4, 'norm_groups': 4, 'dconv_mode': 3, 'dconv_depth': 2, 'dconv_comp': 8, 'dconv_init': 0.001, 'bottom_channels': 512, 't_layers': 5, 't_hidden_scale': 4.0, 't_heads': 8, 't_dropout': 0.02, 't_layer_scale': True, 't_gelu': True, 't_emb': 'sin', 't_max_positions': 10000, 't_max_period': 10000.0, 't_weight_pos_embed': 1.0, 't_cape_mean_normalize': True, 't_cape_augment': True, 't_cape_glob_loc_scale': [5000.0, 1.0, 1.4], 't_sin_random_shift': 0, 't_norm_in': True, 't_norm_in_group': False, 't_group_norm': False, 't_norm_first': True, 't_norm_out': True, 't_weight_decay': 0.0, 't_lr': None, 't_sparse_self_attn': False, 't_sparse_cross_attn': False, 't_mask_type': 'diag', 't_mask_random_seed': 42, 't_sparse_attn_window': 400, 't_global_window': 100, 't_sparsity': 0.95, 't_auto_sparsity': False, 't_cross_first': False, 'rescale': 0.1}

    student_kwargs = {k: v for k, v in teacher_kwargs.items()}
    student_kwargs['channels'] = 12 # 48
    student_kwargs['time_stride'] = 2 # 2
    student_kwargs['t_layers'] = 5 # 5
    student_kwargs['bottom_channels'] = 256 # 512
    student_model = HTDemucs(**student_kwargs)
    student_model.use_train_segment = False
    return student_model

def get_student_teacher_models(partial_weight_copy=True):
    teacher_model = get_teacher_model()
    student_model = get_student_model()
    teacher_model.eval()
    if partial_weight_copy:
        transfer_weights(teacher_model, student_model)
    return student_model, teacher_model