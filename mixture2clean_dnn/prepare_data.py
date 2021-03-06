"""
Summary:  Prepare data.
Author:   Qiuqiang Kong
Created:  2017.12.22
Modified: -
"""
import os
import soundfile
import numpy as np
import argparse
import csv
import time
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import h5py
import librosa
from sklearn import preprocessing
from pathlib import PurePath
from tqdm import tqdm

import prepare_data as pp_data
import config as cfg

from utils import wav_paths, all_file_paths

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs

def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)


    
###
def create_mixture_csv(workspace, speech_dir, noise_dir, data_type,
                       speech_percent, magnification, force=False):
    """Create csv containing mixture information.
    Each line in the .csv file contains [speech_name, noise_name, noise_onset, noise_offset]

    Args:
      workspace: str, path of workspace.
      speech_dir: str, path of speech data.
      noise_dir: str, path of noise data.
      data_type: str, 'train' | 'test'.
      magnification: int, only used when data_type='train', number of noise
          selected to mix with a speech. E.g., when magnication=3, then 4620
          speech with create 4620*3 mixtures. magnification should not larger
          than the species of noises.
    """
    fs = cfg.sample_rate

    out_csv_path = os.path.join(workspace, "mixture_csvs", "%s.csv" % data_type)
    
    if os.path.isfile(out_csv_path) and not force:
        print(f'Mixture CSV {out_csv_path} already exists')
        return
    
    speech_names = wav_paths(speech_dir)
    if speech_percent < 100:
        n_speechs = int(len(speech_names) * speech_percent / 100)
        speech_names = speech_names[:n_speechs]
        
    noise_names = wav_paths(noise_dir)

    rs = np.random.RandomState(0)
    pp_data.create_folder(os.path.dirname(out_csv_path))

    cnt = 0
    f = open(out_csv_path, 'w')
    f.write("%s\t%s\t%s\t%s\n" % ("speech_name", "noise_name", "noise_onset", "noise_offset"))
    for speech_path in tqdm(speech_names, f'Creating mixture CSV ({data_type})'):
        # Read speech.
        speech_na = str(PurePath(speech_path).relative_to(speech_dir))
        (speech_audio, _) = read_audio(speech_path)
        len_speech = len(speech_audio)

        # For training data, mix each speech with randomly picked #magnification noises.
        if data_type == 'train':
            selected_noise_names = rs.choice(noise_names, size=magnification, replace=False)
        # For test data, mix each speech with all noises.
        elif data_type == 'test':
            selected_noise_names = noise_names
        else:
            raise Exception("data_type must be train | test!")

        # Mix one speech with different noises many times.
        for noise_path in selected_noise_names:
            noise_na = str(PurePath(noise_path).relative_to(noise_dir))
            #noise_path = os.path.join(noise_dir, noise_na)
            (noise_audio, _) = read_audio(noise_path)

            len_noise = len(noise_audio)

            if len_noise <= len_speech:
                noise_onset = 0
                noise_offset = len_speech
            # If noise longer than speech then randomly select a segment of noise.
            else:
                noise_onset = rs.randint(0, len_noise - len_speech, size=1)[0]
                noise_offset = noise_onset + len_speech

            cnt += 1
            f.write("%s\t%s\t%d\t%d\n" % (speech_na, noise_na, noise_onset, noise_offset))
    f.close()
    print(out_csv_path)
    print("Create %s mixture csv finished!" % data_type)

###
def calculate_mixture_features(workspace, speech_dir, noise_dir, data_type,
                               snr, force=False):
    """Calculate spectrogram for mixed, speech and noise audio. Then write the
    features to disk.

    Args:
      workspace: str, pa/asteroid/asteroid/4th of workspace.
      speech_dir: str, path of speech data.
      noise_dir: str, path of noise data.
      data_type: str, 'train' | 'test'.
      snr: float, signal to noise ratio to be mixed.
    """
    fs = cfg.sample_rate

    # Open mixture csv.
    mixture_csv_path = os.path.join(workspace, "mixture_csvs", "%s.csv" % data_type)
    with open(mixture_csv_path, newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        lis = list(reader)

    noise_cache = {}
    for i1 in tqdm(range(1, len(lis)), f'Calculating mixture features ({data_type})'):
        [speech_na, noise_na, noise_onset, noise_offset] = lis[i1]
        noise_onset = int(noise_onset)
        noise_offset = int(noise_offset)

        # Construct the output paths and see if the features/mixed audio are already computed
        speech_path = os.path.join(speech_dir, speech_na)
        noise_path = os.path.join(noise_dir, noise_na)

        out_bare_na = os.path.join("%s.%s" %
            (os.path.splitext(speech_na)[0], os.path.splitext(noise_na)[0]))
        out_audio_path = os.path.join(workspace, "mixed_audios", "spectrogram",
            data_type, "%ddb" % int(snr), "%s.wav" % out_bare_na)
        
        out_feat_path = os.path.join(workspace, "features", "spectrogram",
            data_type, "%ddb" % int(snr), "%s.p" % out_bare_na)
        
        if os.path.isfile(out_audio_path) and os.path.isfile(out_feat_path) and not force:
            print(f'Mixed audio {out_audio_path} and its features are already computed')
            continue
        
        # Read speech audio and noise audio
        (speech_audio, _) = read_audio(speech_path, target_fs=fs)
        
        try:
            noise_audio = noise_cache[noise_path]
        except KeyError:
            (noise_audio, _) = read_audio(noise_path, target_fs=fs)
            noise_cache[noise_path] = noise_audio

        # Repeat noise to the same length as speech.
        orig_noise_len = len(noise_audio)
        if len(noise_audio) < len(speech_audio):
            n_repeat = int(np.ceil(float(len(speech_audio)) / float(len(noise_audio))))
            noise_audio_ex = np.tile(noise_audio, n_repeat)
            noise_audio = noise_audio_ex[0 : len(speech_audio)]
        # Truncate noise to the same length as speech.
        else:
            if noise_offset - noise_onset == len(speech_audio):       # a hacky workaround around a weird bug
                if noise_offset > len(noise_audio):
                    dif = noise_offset - len(noise_audio)
                    noise_offset -= dif
                    noise_onset -= dif
                    
                noise_audio = noise_audio[noise_onset : noise_offset]
            else:
                noise_audio = noise_audio[:len(speech_audio)]
                
        if len(noise_audio) != len(speech_audio):
            print("noise len {}, orig {}, speech len {}".format(len(noise_audio), orig_noise_len, len(speech_audio)))
            print("onset {}, offset {}".format(noise_onset, noise_offset))
            raise ValueError("Stupid lenghts!")
            
        clean_rms = cal_rms(speech_audio)
        noise_rms = cal_rms(noise_audio)
        adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)
        
        # Scale noise to given snr
        noise_audio = noise_audio * (adjusted_noise_rms / noise_rms)

        # Get normalized mixture, speech, noise.
        (mixed_audio, speech_audio, noise_audio, alpha) = additive_mixing(speech_audio, noise_audio)

        # Write out mixed audio.
        create_folder(os.path.dirname(out_audio_path))
        write_audio(out_audio_path, mixed_audio, fs)

        # Extract spectrogram.
        mixed_complx_x = calc_sp(mixed_audio, mode='complex')
        speech_x = calc_sp(speech_audio, mode='magnitude')
        noise_x = calc_sp(noise_audio, mode='magnitude')
        ir_mask = np.minimum(speech_x / np.abs(mixed_complx_x), 1)

        # Write out features.
        create_folder(os.path.dirname(out_feat_path))
        data = [mixed_complx_x, speech_x, noise_x, ir_mask, alpha, out_bare_na]
        pickle.dump(data, open(out_feat_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10 ** a)
    return noise_rms

def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))

def additive_mixing(s, n):
    """Mix normalized source1 and source2.

    Args:
      s: ndarray, source1.
      n: ndarray, source2.

    Returns:
      mix_audio: ndarray, mixed audio.
      s: ndarray, pad or truncated and scalered source1.
      n: ndarray, scaled source2.
      alpha: float, normalize coefficient.
    """
    mixed_amp = s + n
    alpha = 1.0
    
    if mixed_amp.max(axis=0) > 1 or mixed_amp.min(axis=0) < -1:
        if mixed_amp.max(axis=0) >= abs(mixed_amp.min(axis=0)):
            alpha = 1. / mixed_amp.max(axis=0)
        else:
            alpha = -1. / mixed_amp.min(axis=0)
        mixed_amp = mixed_amp * alpha
        s *= alpha
        n *= alpha

    return mixed_amp, s, n, alpha

def calc_sp(audio, mode):
    """Calculate spectrogram.

    Args:
      audio: 1darray.
      mode: string, 'magnitude' | 'complex'

    Returns:
      spectrogram: 2darray, (n_time, n_freq).
    """
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    ham_win = np.hamming(n_window)
    [f, t, x] = signal.spectral.spectrogram(
                    audio,
                    window=ham_win,
                    nperseg=n_window,
                    noverlap=n_overlap,
                    detrend=False,
                    return_onesided=True,
                    mode=mode)
    x = x.T
    if mode == 'magnitude':
        x = x.astype(np.float32)
    elif mode == 'complex':
        x = x.astype(np.complex64)
    else:
        raise Exception("Incorrect mode!")
    return x

###
def pack_features(workspace, data_type, snr, n_concat,
                  n_hop, force=False):
    """Load all features, apply log and conver to 3D tensor, write out to .h5 file.

    Args:
      workspace: str, path of workspace.
      data_type: str, 'train' | 'test'.
      snr: float, signal to noise ratio to be mixed.
      n_concat: int, number of frames to be concatenated.
      n_hop: int, hop frames.
    """
    
    out_path = os.path.join(workspace, "packed_features", "spectrogram", data_type, "%ddb" % int(snr), "data.h5")
    if os.path.isfile(out_path) and not force:
        print(f'Packed features file {out_path} is already made')
        return
    
    x_all = []  # (n_segs, n_concat, n_freq)
    y_all = []  # (n_segs, n_freq)

    t1 = time.time()

    # Load all features.
    feat_dir = os.path.join(workspace, "features", "spectrogram", data_type, "%ddb" % int(snr))
    feat_paths = all_file_paths(feat_dir)
    
    # Read the spectrograms and count their lengths to pre-allocate arrays
    n_segs = 0
    for feat_path in tqdm(feat_paths, 'Reading features first time (segments lenghts estimation)'):
        data = pickle.load(open(feat_path, 'rb'))
        [mixed_complx_x, _, _, _, _, _] = data
        sgs = (mixed_complx_x.shape[0] + n_concat - 1) // n_hop
        n_segs += sgs
        
    print(f'Upper estimate for the number of segmens: {n_segs}')
    
    n_freq = cfg.n_window // 2 + 1
    x_all = np.zeros((n_segs, n_concat, n_freq), dtype=np.float32)
    y_all = np.zeros((n_segs, n_freq), dtype=np.float32)
    
    count_segs = 0
    for feat_path in tqdm(feat_paths, 'Packing features'):
        # Load feature.
        na = str(PurePath(feat_path).relative_to(feat_dir))
        data = pickle.load(open(feat_path, 'rb'))
        [mixed_complx_x, speech_x, noise_x, ir_mask, alpha, na] = data
        mixed_x = np.abs(mixed_complx_x)

        # Pad start and finish of the spectrogram with border values.
        n_pad = (n_concat - 1) // 2
        mixed_x = pad_with_border(mixed_x, n_pad)
        
        # For training, we pack ideal ratio masks
        # speech_x = pad_with_border(speech_x, n_pad)
        ir_mask = pad_with_border(ir_mask, n_pad)

        # Cut input spectrogram to 3D segments with n_concat.
        mixed_x_3d = mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=n_hop)
        x = log_sp(mixed_x_3d).astype(np.float32)
        x_all[count_segs:count_segs+len(x), :, :] = x

        # Cut target spectrogram and take the center frame of each 3D segment.
        ir_mask_3d = mat_2d_to_3d(ir_mask, agg_num=n_concat, hop=n_hop)
        y = ir_mask_3d[:, n_pad, :].astype(np.float32)
        #y = log_sp(y).astype(np.float32)
        y_all[count_segs:count_segs+len(y), :] = y

        assert len(x) == len(y)
        count_segs += len(x)

    x_all = x_all[:count_segs]
    y_all = y_all[:count_segs]

    # Write out data to .h5 file.
    create_folder(os.path.dirname(out_path))
    with h5py.File(out_path, 'w') as hf:
        hf.create_dataset('x', data=x_all)
        hf.create_dataset('y', data=y_all)

    print("Write out to %s" % out_path)
    print("Pack features finished! %s s" % (time.time() - t1,))


def log_sp(x):
    return np.log(x + 1e-08)


def mat_2d_to_3d(x, agg_num, hop):
    """Segment 2D array to 3D segments.
    """
    # Pad to at least one block.
    len_x, n_in = x.shape
    if (len_x < agg_num):
        x = np.concatenate((x, np.zeros((agg_num - len_x, n_in))))

    # Segment 2d to 3d.
    len_x = len(x)
    i1 = 0
    x3d = []
    while (i1 + agg_num <= len_x):
        x3d.append(x[i1 : i1 + agg_num])
        i1 += hop
    return np.array(x3d)


def pad_with_border(x, n_pad):
    """Pad the begin and finish of spectrogram with border frame value.
    """
    n_pad = int(n_pad)
    return np.pad(x, ((n_pad, n_pad), (0, 0)), mode='edge')
    

def compute_scaler(workspace, data_type, snr, force=False):
    """Compute and write out scaler of data.
    """

    # Check if already computed
    out_path = os.path.join(workspace, "packed_features", "spectrogram", data_type, "%ddb" % int(snr), "scaler.p")
    if os.path.isfile(out_path) and not force:
        print(f'Scaler for {data_type} is already computed')
        return
    
    # Load data.
    print(f'Computing scaler {data_type}')
    t1 = time.time()
    hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", data_type, "%ddb" % int(snr), "data.h5")
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf.get('x')
        x = np.array(x)     # (n_segs, n_concat, n_freq)

    # Compute scaler.
    (n_segs, n_concat, n_freq) = x.shape
    x2d = x.reshape((n_segs * n_concat, n_freq))
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(x2d)
    print(scaler.mean_)
    print(scaler.scale_)

    # Write out scaler.
    create_folder(os.path.dirname(out_path))
    pickle.dump(scaler, open(out_path, 'wb'))

    print("Save scaler to %s" % out_path)
    print("Compute scaler finished! %s s" % (time.time() - t1,))

    
def combine_scalers(sc1, sc2):
    """
    Combines scalers for two subsets of data
    """
    assert sc1.n_features_in_ == sc2.n_features_in_
    
    N = sc1.n_samples_seen_ + sc2.n_samples_seen_
    mean = (sc1.mean_ * sc1.n_samples_seen_ + sc2.mean_ * sc2.n_samples_seen_) / N
    
    squares_sum1 = (sc1.var_ + sc1.mean_**2) * sc1.n_samples_seen_
    squares_sum2 = (sc2.var_ + sc2.mean_**2) * sc2.n_samples_seen_
    var = (squares_sum1 + squares_sum2) / N - mean**2
    
    std = np.sqrt(var)
    new_scaler = preprocessing.StandardScaler()
    new_scaler.mean_ = mean
    new_scaler.var_ = var
    new_scaler.scale_ = std
    new_scaler.n_samples_seen_ = N
    new_scaler.n_features_in_ = sc1.n_features_in_
    
    return new_scaler
    
    
def scale_on_2d(x2d, scaler):
    """Scale 2D array data.
    """
    return scaler.transform(x2d)

def scale_on_3d(x3d, scaler):
    """Scale 3D array data.
    """
    (n_segs, n_concat, n_freq) = x3d.shape
    x2d = x3d.reshape((n_segs * n_concat, n_freq))
    x2d = scaler.transform(x2d)
    x3d = x2d.reshape((n_segs, n_concat, n_freq))
    return x3d

def inverse_scale_on_2d(x2d, scaler):
    """Inverse scale 2D array data.
    """
    return x2d * scaler.scale_[None, :] + scaler.mean_[None, :]

###
def load_hdf5(hdf5_path):
    """Load hdf5 data.
    """
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf.get('x')
        y = hf.get('y')
        x = np.array(x)     # (n_segs, n_concat, n_freq)
        y = np.array(y)     # (n_segs, n_freq)
    return x, y

def np_mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

###
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_create_mixture_csv = subparsers.add_parser('create_mixture_csv')
    parser_create_mixture_csv.add_argument('--workspace', type=str, required=True)
    parser_create_mixture_csv.add_argument('--speech_dir', type=str, required=True)
    parser_create_mixture_csv.add_argument('--noise_dir', type=str, required=True)
    parser_create_mixture_csv.add_argument('--data_type', type=str, required=True)
    parser_create_mixture_csv.add_argument('--speech_percent', type=int, default=100)
    parser_create_mixture_csv.add_argument('--magnification', type=int, default=1)
    parser_create_mixture_csv.add_argument('--force', action='store_true')

    parser_calculate_mixture_features = subparsers.add_parser('calculate_mixture_features')
    parser_calculate_mixture_features.add_argument('--workspace', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--speech_dir', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--noise_dir', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--data_type', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--snr', type=float, required=True)
    parser_calculate_mixture_features.add_argument('--force', action='store_true')
    
    parser_pack_features = subparsers.add_parser('pack_features')
    parser_pack_features.add_argument('--workspace', type=str, required=True)
    parser_pack_features.add_argument('--data_type', type=str, required=True)
    parser_pack_features.add_argument('--snr', type=float, required=True)
    parser_pack_features.add_argument('--n_concat', type=int, required=True)
    parser_pack_features.add_argument('--n_hop', type=int, required=True)
    parser_pack_features.add_argument('--force', action='store_true')

    parser_compute_scaler = subparsers.add_parser('compute_scaler')
    parser_compute_scaler.add_argument('--workspace', type=str, required=True)
    parser_compute_scaler.add_argument('--data_type', type=str, required=True)
    parser_compute_scaler.add_argument('--snr', type=float, required=True)
    parser_compute_scaler.add_argument('--force', action='store_true')

    args = parser.parse_args()
    kwargs = vars(args).copy()
    del kwargs['mode']
    
    if args.mode == 'create_mixture_csv':
        create_mixture_csv(**kwargs)
    elif args.mode == 'calculate_mixture_features':
        calculate_mixture_features(**kwargs)
    elif args.mode == 'pack_features':
        pack_features(**kwargs)
    elif args.mode == 'compute_scaler':
        compute_scaler(**kwargs)
    else:
        raise Exception("Error!")
