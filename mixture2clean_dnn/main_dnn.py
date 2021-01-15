"""
Summary:  Train, inference and evaluate speech enhancement.
Author:   Qiuqiang Kong
Created:  2017.12.22
Modified: -
"""
import numpy as np
import os
import pickle
import h5py
import argparse
import time
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import prepare_data as pp_data
import config as cfg
from pathlib import PurePath
from tqdm import tqdm

from data_generator import DataGenerator
from spectrogram_to_wave import recover_wav
from utils import all_file_paths
from prepare_data import combine_scalers

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model


def eval(model, gen, x, y):
    """Validation function.

    Args:
      model: keras model.
      gen: object, data generator.
      x: 3darray, input, (n_segs, n_concat, n_freq)
      y: 2darray, target, (n_segs, n_freq)
    """
    pred_all, y_all = [], []

    # Inference in mini batch.
    for (batch_x, batch_y) in gen.generate(xs=[x], ys=[y]):
        pred = model.predict(batch_x)
        pred_all.append(pred)
        y_all.append(batch_y)

    # Concatenate mini batch prediction.
    pred_all = np.concatenate(pred_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    # Compute loss.
    loss = pp_data.np_mean_absolute_error(y_all, pred_all)
    return loss


def read_combined_scaler(workspace, tr_snr):
    scaler = None
    for snr in tr_snr:
        scaler_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "%ddb" % int(snr), "scaler.p")
        sc = pickle.load(open(scaler_path, 'rb'))
        if scaler is None:
            scaler = sc
        else:
            scaler = combine_scalers(scaler, sc)
    return scaler


def train(workspace, tr_snr, te_snr, lr, model_name=None, force=False, iters=100000):
    """Train the neural network. Write out model every several iterations.

    Args:
      workspace: str, path of workspace.
      tr_snr: float, training SNR.
      te_snr: float, testing SNR.
      lr: float, learning rate.
    """
    
    # Directories for saving models and training stats
    if model_name is None:
        model_name = '_'.join([str(snr) for snr in tr_snr]) + 'ddbs'
    
    model_dir = os.path.join(workspace, "models", model_name)
    pp_data.create_folder(model_dir)

    stats_dir = os.path.join(workspace, "training_stats", model_name)
    pp_data.create_folder(stats_dir)
    
    model_path = os.path.join(model_dir, f"md_{iters}iters.h5")
    if os.path.isfile(model_path) and not force:
        print(f'Model already trained ({model_path})')
        return
    
    # Load data.
    t1 = time.time()
    tr_x = None
    tr_y = None
    
    for snr in tr_snr:  
        tr_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "%ddb" % int(snr), "data.h5")
        (X, y) = pp_data.load_hdf5(tr_hdf5_path)
        if tr_x is None:
            tr_x = X
            tr_y = y
        else:
            tr_x = np.concatenate((tr_x, X))
            tr_y = np.concatenate((tr_y, y))
    
    te_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "test", "%ddb" % int(te_snr), "data.h5")
    (te_x, te_y) = pp_data.load_hdf5(te_hdf5_path)
    print(tr_x.shape, tr_y.shape)
    print(te_x.shape, te_y.shape)
    print("Load data time: %s s" % (time.time() - t1,))

    batch_size = 500
    print("%d iterations / epoch" % int(tr_x.shape[0] / batch_size))

    # Scale data.
    if True:
        t1 = time.time()
        scaler = read_combined_scaler(workspace, tr_snr)
            
        tr_x = pp_data.scale_on_3d(tr_x, scaler)
        #tr_y = pp_data.scale_on_2d(tr_y, scaler)
        te_x = pp_data.scale_on_3d(te_x, scaler)
        #te_y = pp_data.scale_on_2d(te_y, scaler)
        print("Scale data time: %s s" % (time.time() - t1,))

    # Debug plot.
    if False:
        plt.matshow(tr_x[0 : 1000, 0, :].T, origin='lower', aspect='auto', cmap='jet')
        plt.show()
        pause

    print(tf.test.is_gpu_available())

    # Build model
    (_, n_concat, n_freq) = tr_x.shape
    n_hid = 2048

    model = Sequential()
    model.add(Flatten(input_shape=(n_concat, n_freq)))
    model.add(Dense(n_hid, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_hid, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_hid, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_freq, activation='linear'))
    model.summary()

    model.compile(loss='mean_absolute_error',
                  optimizer=Adam(lr=lr))

    # Data generator.
    tr_gen = DataGenerator(batch_size=batch_size, type='train')
    eval_te_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=50)
    eval_tr_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=50)

    # Print loss before training.
    iter = 0
    tr_loss = eval(model, eval_tr_gen, tr_x, tr_y)
    te_loss = eval(model, eval_te_gen, te_x, te_y)
    print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iter, tr_loss, te_loss))

    # Save out training stats.
    stat_dict = {'iter': iter,
                    'tr_loss': tr_loss,
                    'te_loss': te_loss, }
    stat_path = os.path.join(stats_dir, "%diters.p" % iter)
    pickle.dump(stat_dict, open(stat_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    # Train.h
    t1 = time.time()
    for (batch_x, batch_y) in tr_gen.generate(xs=[tr_x], ys=[tr_y]):
        loss = model.train_on_batch(batch_x , batch_y)
        iter += 1

        # Validate and save training stats.
        if iter % 1000 == 0:
            tr_loss = eval(model, eval_tr_gen, tr_x, tr_y)
            te_loss = eval(model, eval_te_gen, te_x, te_y)
            print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iter, tr_loss, te_loss))

            # Save out training stats.
            stat_dict = {'iter': iter,
                         'tr_loss': tr_loss,
                         'te_loss': te_loss, }
            stat_path = os.path.join(stats_dir, "%diters.p" % iter)
            pickle.dump(stat_dict, open(stat_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        # Save model.
        if iter % 10000 == 0:
            model_path = os.path.join(model_dir, "md_%diters.h5" % iter)
            model.save(model_path)
            print("Saved model to %s" % model_path)

        if iter == iters+1:
            break

    print("Training time: %s s" % (time.time() - t1,))

def inference(workspace, tr_snr, te_snr, n_concat,
              iteration, model_name=None, visualize=False, force=False):
    """Inference all test data, write out recovered wavs to disk.

    Args:
      workspace: str, path of workspace.
      tr_snr: float, training SNR.
      te_snr: float, testing SNR.
      n_concat: int, number of frames to concatenta, should equal to n_concat
          in the training stage.
      iter: int, iteration of model to load.
      visualize: bool, plot enhanced spectrogram for debug.
    """
    
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    fs = cfg.sample_rate
    scale = True

    if model_name is None:
        model_name = '_'.join([str(snr) for snr in tr_snr]) + 'ddbs'
    
    # Load model.
    model_path = os.path.join(workspace, "models", model_name, "md_%diters.h5" % iteration)
    print('GPU available: ', tf.test.is_gpu_available())
    
    model = load_model(model_path)

    # Load scaler.
    scaler = read_combined_scaler(workspace, tr_snr)

    for snr in te_snr:
        # Load test data.
        feat_dir = os.path.join(workspace, "features", "spectrogram", "test", "%ddb" % int(snr))
        feat_paths = all_file_paths(feat_dir)

        for (cnt, feat_path) in tqdm(enumerate(feat_paths), 'Inference (creating enhanced speech)'):
            # Check if the enhanced audio is already inferred
            na = str(PurePath(feat_path).relative_to(feat_dir).with_suffix(''))
            out_path = os.path.join(workspace, "enh_wavs", "test", model_name, "%ddb" % int(snr), "%s.enh.wav" % na)
            if os.path.isfile(out_path) and not force:
                print(f'Enhanced audio {out_path} is already made')
                continue

            # Load feature.
            data = pickle.load(open(feat_path, 'rb'))
            [mixed_cmplx_x, speech_x, noise_x, ir_mask, alpha, na] = data
            mixed_x = np.abs(mixed_cmplx_x)

            # Process data.
            n_pad = (n_concat - 1) / 2
            mixed_x = pp_data.pad_with_border(mixed_x, n_pad)
            mixed_x = pp_data.log_sp(mixed_x)
            speech_x = pp_data.log_sp(speech_x)

            # Scale data.
            if scale:
                mixed_x = pp_data.scale_on_2d(mixed_x, scaler)
                speech_x = pp_data.scale_on_2d(speech_x, scaler)

            # Cut input spectrogram to 3D segments with n_concat.
            mixed_x_3d = pp_data.mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=1)

            # Predict.
            pred = model.predict(mixed_x_3d)
            #print(cnt, na)

            # Inverse scale.
            if scale:
                mixed_x = pp_data.inverse_scale_on_2d(mixed_x, scaler)
                speech_x = pp_data.inverse_scale_on_2d(speech_x, scaler)
                #pred = pp_data.inverse_scale_on_2d(pred, scaler)

            # Debug plot.
            if visualize:
                fig, axs = plt.subplots(3,1, sharex=False)
                axs[0].matshow(mixed_x.T, origin='lower', aspect='auto', cmap='jet')
                axs[1].matshow(speech_x.T, origin='lower', aspect='auto', cmap='jet')
                axs[2].matshow(pred.T, origin='lower', aspect='auto', cmap='jet')
                axs[0].set_title("%ddb mixture log spectrogram" % int(te_snr))
                axs[1].set_title("Clean speech log spectrogram")
                axs[2].set_title("Enhanced speech log spectrogram")
                for j1 in xrange(3):
                    axs[j1].xaxis.tick_bottom()
                plt.tight_layout()
                plt.show()

            # Recover enhanced wav
            s = recover_wav(pred, mixed_cmplx_x, n_overlap, np.hamming, irr_mask=True)
            s *= np.sqrt((np.hamming(n_window)**2).sum())   # Scaler for compensate the amplitude
                                                            # change after spectrogram and IFFT.

            # Write out enhanced wav.
            pp_data.create_folder(os.path.dirname(out_path))
            pp_data.write_audio(out_path, s, fs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--model_name', type=str, default=None)
    parser_train.add_argument('--tr_snr', type=float, nargs='+', required=True)
    parser_train.add_argument('--te_snr', type=float, required=True)
    parser_train.add_argument('--lr', type=float, required=True)
    parser_train.add_argument('--force', action='store_true')

    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--workspace', type=str, required=True)
    parser_inference.add_argument('--model_name', type=str)
    parser_inference.add_argument('--tr_snr', type=float, nargs='+', required=True)
    parser_inference.add_argument('--te_snr', type=float, nargs='+', required=True)
    parser_inference.add_argument('--n_concat', type=int, required=True)
    parser_inference.add_argument('--iteration', type=int, required=True)
    parser_inference.add_argument('--force', action='store_true')

    args = parser.parse_args()
    kwargs = vars(args).copy()
    del kwargs['mode']
    
    if args.mode == 'train':
        train(**kwargs)
    elif args.mode == 'inference':
        inference(**kwargs)
    else:
        raise Exception("Error!")
