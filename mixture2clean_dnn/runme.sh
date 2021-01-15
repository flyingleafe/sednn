#!/bin/bash

FORCE=''

if [ "$1" == "--force" ]; then
  FORCE='--force'
fi

MINIDATA=0
if [ $MINIDATA -eq 1 ]; then
  WORKSPACE="workspace"
  mkdir $WORKSPACE
  TR_SPEECH_DIR="mini_data/train_speech"
  TR_NOISE_DIR="mini_data/train_noise"
  TE_SPEECH_DIR="mini_data/test_speech"
  TE_NOISE_DIR="mini_data/test_noise"
  TEST_SPEECH_PERCENT=100
  echo "Using mini data. "
else
  WORKSPACE="/import/vision-eddydata/dm005_tmp"
  TR_SPEECH_DIR="$HOME/datasets/TIMIT/data/TRAIN"
  TR_NOISE_DIR="$HOME/datasets/noises-train"
  TE_SPEECH_DIR="$HOME/datasets/TIMIT/data/TEST"
  TE_NOISE_DIR="$HOME/datasets/noises-test-drones"
  TEST_SPEECH_PERCENT=100
  echo "Using full data. "
fi

# Create mixture csv.
python prepare_data.py create_mixture_csv --workspace=$WORKSPACE --speech_dir=$TR_SPEECH_DIR --noise_dir=$TR_NOISE_DIR --data_type=train --magnification=2 $FORCE
python prepare_data.py create_mixture_csv --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --noise_dir=$TE_NOISE_DIR --data_type=test --speech_percent=$TEST_SPEECH_PERCENT $FORCE

TR_SNRs=( -25 -20 -15 -10 -5 0 5 10 15 )
TE_SNRs=( -30 -25 -20 -15 -10 -5 0 )

N_CONCAT=7
N_HOP=3

for TR_SNR in "${TR_SNRs[@]}"; do
    echo "train SNR: $TR_SNR db"
    python prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$TR_SPEECH_DIR --noise_dir=$TR_NOISE_DIR --data_type=train --snr=$TR_SNR $FORCE
    
    python prepare_data.py pack_features --workspace=$WORKSPACE --data_type=train --snr=$TR_SNR --n_concat=$N_CONCAT --n_hop=$N_HOP $FORCE
    
    # Compute scaler.
    python prepare_data.py compute_scaler --workspace=$WORKSPACE --data_type=train --snr=$TR_SNR $FORCE    
done

for TE_SNR in "${TE_SNRs[@]}"; do
    echo "test SNR: $TE_SNR db"
    python prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --noise_dir=$TE_NOISE_DIR --data_type=test --snr=$TE_SNR $FORCE
    
    python prepare_data.py pack_features --workspace=$WORKSPACE --data_type=test --snr=$TE_SNR --n_concat=$N_CONCAT --n_hop=$N_HOP $FORCE
done

# for SNR in "${SNRs[@]}"; do
#     echo "SNR: $SNR db"
#     # Calculate mixture features.
#     python prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$TR_SPEECH_DIR --noise_dir=$TR_NOISE_DIR --data_type=train --snr=$SNR $FORCE
    

#     # Pack features.
    

    
    
#     LEARNING_RATE=1e-4 
#     python main_dnn.py train --workspace=$WORKSPACE --tr_snr=$SNR --te_snr=$SNR --lr=$LEARNING_RATE $FORCE

#     # Plot training stat.
#     python evaluate.py plot_training_stat --workspace=$WORKSPACE --tr_snr=$SNR --bgn_iter=0 --fin_iter=10001 --interval_iter=1000

#     # Inference, enhanced wavs will be created.
#     ITERATION=10000
#     python main_dnn.py inference --workspace=$WORKSPACE --tr_snr=$SNR --te_snr=$SNR --n_concat=$N_CONCAT --iteration=$ITERATION $FORCE

#     # Calculate PESQ of all enhanced speech.
#     python evaluate.py calculate_pesq --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --te_snr=$SNR $FORCE

#     # Calculate overall stats.
#     python evaluate.py get_stats --workspace=$WORKSPACE --te_snr=$SNR
# done

#TR_SNR=0
#TE_SNR=0

# Train.
# LEARNING_RATE=1e-4 
# python main_dnn.py train --workspace=$WORKSPACE --tr_snr=$TR_SNR --te_snr=$TE_SNR --lr=$LEARNING_RATE $FORCE

# # Plot training stat.
# python evaluate.py plot_training_stat --workspace=$WORKSPACE --tr_snr=$TR_SNR --bgn_iter=0 --fin_iter=10001 --interval_iter=1000

# # Inference, enhanced wavs will be created.
# ITERATION=10000
# python main_dnn.py inference --workspace=$WORKSPACE --tr_snr=$TR_SNR --te_snr=$TE_SNR --n_concat=$N_CONCAT --iteration=$ITERATION $FORCE

# # Calculate PESQ of all enhanced speech.
# python evaluate.py calculate_pesq --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --te_snr=$TE_SNR $FORCE

# # Calculate overall stats.
# python evaluate.py get_stats --workspace=$WORKSPACE
