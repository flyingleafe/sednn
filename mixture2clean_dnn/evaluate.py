"""
Summary:  Calculate PESQ and overal stats of enhanced speech.
Author:   Qiuqiang Kong
Created:  2017.12.22
Modified: -
"""
import argparse
import os
import csv
import numpy as np
import pandas as pd
import soundfile as sf
import pickle
import matplotlib.pyplot as plt

from pypesq import pesq
from pathlib import PurePath
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from prepare_data import read_audio
from utils import all_file_paths

def plot_training_stat(workspace, tr_snr, bgn_iter, fin_iter, interval_iter):
    """Plot training and testing loss.

    Args:
      workspace: str, path of workspace.
      tr_snr: float, training SNR.
      bgn_iter: int, plot from bgn_iter
      fin_iter: int, plot finish at fin_iter
      interval_iter: int, interval of files.
    """
    tr_losses, te_losses, iters = [], [], []

    # Load stats.
    stats_dir = os.path.join(workspace, "training_stats", "%ddb" % int(tr_snr))
    for iter in range(bgn_iter, fin_iter, interval_iter):
        stats_path = os.path.join(stats_dir, "%diters.p" % iter)
        dict = pickle.load(open(stats_path, 'rb'))
        tr_losses.append(dict['tr_loss'])
        te_losses.append(dict['te_loss'])
        iters.append(dict['iter'])

    # Plot
    line_tr, = plt.plot(tr_losses, c='b', label="Train")
    line_te, = plt.plot(te_losses, c='r', label="Test")
    plt.axis([0, len(iters), 0, max(tr_losses)])
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(handles=[line_tr, line_te])
    plt.xticks(np.arange(len(iters)), iters)
    plt.show()


def calculate_pesq(workspace, speech_dir, model_name, te_snr, calc_mixed=False, force=False):
    """Calculate PESQ of all enhaced speech.

    Args:
      workspace: str, path of workspace.
      speech_dir: str, path of clean speech.
      te_snr: float, testing SNR.
    """
    results_file = os.path.join(workspace, 'evaluation', f'pesq_results_{model_name}.csv')
    if os.path.isfile(results_file) and not force:
        print('PESQ is already calculated')
        return pd.read_csv(results_file)
    
    df = pd.DataFrame(columns=['filepath', 'snr', 'pesq'])
    speech_audio_cache = {}
    
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
    for snr in te_snr:
        print(f'SNR: {snr}')

        # Calculate PESQ of all enhaced speech.
        if calc_mixed:
            enh_speech_dir = os.path.join(workspace, "mixed_audios", "spectrogram", "test", "%ddb" % int(snr))
        else:
            enh_speech_dir = os.path.join(workspace, "enh_wavs", "test", model_name, "%ddb" % int(snr))
        
        enh_paths = all_file_paths(enh_speech_dir)

        pendings = []
        with ProcessPoolExecutor(10) as pool:
            for (cnt, enh_path) in tqdm(enumerate(enh_paths), 'Calculating PESQ score (submitting)'):
                # enh_path = os.path.join(enh_speech_dir, na)
                na = str(PurePath(enh_path).relative_to(enh_speech_dir))
                #print(cnt, na)

                if calc_mixed:
                    speech_na = '.'.join(na.split('.')[:-2])
                else:
                    speech_na = '.'.join(na.split('.')[:-3])

                speech_path = os.path.join(speech_dir, f"{speech_na}.wav")

                deg, sr = read_audio(enh_path)

                try:
                    ref = speech_audio_cache[speech_path]
                except KeyError:
                    ref, _ = read_audio(speech_path, target_fs=sr)
                    speech_audio_cache[speech_path] = ref

                if len(ref) < len(deg):
                    ref = np.pad(ref, (0, len(deg) - len(ref)))
                elif len(deg) < len(ref):
                    deg = np.pad(deg, (0, len(ref) - len(deg)))

                pendings.append(pool.submit(_calc_pesq, ref, deg, sr, na, snr))
        
            for pending in tqdm(pendings, 'Collecting pending jobs'):
                score, na, snr = pending.result()
                df.loc[len(df)] = [na, snr, score]

        df.to_csv(results_file, index=False)
        
    return df

def _calc_pesq(ref, deg, sr, na, snr):
    score = pesq(ref, deg, sr)
    return score, na, snr

def get_stats(workspace, te_snr):
    """Calculate stats of PESQ.
    """
    df = pd.read_csv(os.path.join(workspace, 'evaluation', f'pesq_results_{te_snr}db.csv'))

    pesq_dict = {}
    for idx, row in df.iterrows():
        na = row[0]
        pesq = float(row[2])
        noise_type = na.split('.')[-3]
        if noise_type not in pesq_dict.keys():
            pesq_dict[noise_type] = [pesq]
        else:
            pesq_dict[noise_type].append(pesq)

    avg_list, std_list = [], []
    f = "{0:<16} {1:<16}"
    print(f.format("Noise", "PESQ"))
    print("---------------------------------")
    for noise_type in pesq_dict.keys():
        pesqs = pesq_dict[noise_type]
        avg_pesq = np.mean(pesqs)
        std_pesq = np.std(pesqs)
        avg_list.append(avg_pesq)
        std_list.append(std_pesq)
        print(f.format(noise_type, "%.2f +- %.2f" % (avg_pesq, std_pesq)))
    print("---------------------------------")
    print(f.format("Avg.", "%.2f +- %.2f" % (np.mean(avg_list), np.mean(std_list))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_plot_training_stat = subparsers.add_parser('plot_training_stat')
    parser_plot_training_stat.add_argument('--workspace', type=str, required=True)
    parser_plot_training_stat.add_argument('--tr_snr', type=float, required=True)
    parser_plot_training_stat.add_argument('--bgn_iter', type=int, required=True)
    parser_plot_training_stat.add_argument('--fin_iter', type=int, required=True)
    parser_plot_training_stat.add_argument('--interval_iter', type=int, required=True)

    parser_calculate_pesq = subparsers.add_parser('calculate_pesq')
    parser_calculate_pesq.add_argument('--workspace', type=str, required=True)
    parser_calculate_pesq.add_argument('--speech_dir', type=str, required=True)
    parser_calculate_pesq.add_argument('--te_snr', type=float, required=True)
    parser_calculate_pesq.add_argument('--force', action='store_true')

    parser_get_stats = subparsers.add_parser('get_stats')
    parser_get_stats.add_argument('--workspace', type=str, required=True)
    parser_get_stats.add_argument('--te_snr', type=float, required=True)
    
    args = parser.parse_args()
    kwargs = vars(args).copy()
    del kwargs['mode']
    
    if args.mode == 'plot_training_stat':
        plot_training_stat(**kwargs)
    elif args.mode == 'calculate_pesq':
        calculate_pesq(**kwargs)
    elif args.mode == 'get_stats':
        get_stats(**kwargs)
    else:
        raise Exception("Error!")
