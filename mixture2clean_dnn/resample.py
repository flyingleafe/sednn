import sys
import os

from tqdm import tqdm
from pathlib import PurePath
from prepare_data import read_audio, write_audio
from utils import wav_paths

def main(from_dir, to_dir, sr):
    from_paths = wav_paths(from_dir)
    for from_p in tqdm(from_paths, 'Resampling audio'):
        rel_p = PurePath(from_p).relative_to(from_dir)
        to_p = to_dir / rel_p
        os.makedirs(to_p.parent, exist_ok=True)

        wav, _ = read_audio(from_p, sr)
        write_audio(to_p, wav, sr)          

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
