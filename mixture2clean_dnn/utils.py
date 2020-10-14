import os

def wav_paths(root_dir):
    return [os.path.join(root, name)
            for root, dirs, files in os.walk(root_dir)
            for name in files if name.endswith(".wav")]
    
def all_file_paths(root_dir):
    return [os.path.join(root, name)
            for root, dirs, files in os.walk(root_dir)
            for name in files]