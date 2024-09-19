import os, fnmatch
from bidict import bidict
import numpy as np
import pandas as pd
import torchaudio
import torchaudio.transforms as transforms
from tqdm import tqdm



class LaberEncoder:
    def __init__ (self, labels, order = None):
        self.map = bidict()
        if order is not None and isinstance(order, np.ndarray):
            if len(order) != len(np.unique(labels)):
                print('The length of \'order\' must be equal to the number of unique items in \'labels\' ');
                return
            else:
                unique_items = order
        else:
            unique_items = np.unique(labels)

        for i, item in enumerate(unique_items):
            self.map[item] = i


    def encode (self, inputs):
        encoded_labels = []
        try:
            for input in inputs:
                encoded_labels.append(self.map[input])
        except KeyError as e:
            print("Error:", e)
        return encoded_labels.copy()


    def decode (self, inputs):
        decoded_labels = []
        try:
            for input in inputs:
                decoded_labels.append(self.map.inverse[input])
        except KeyError as e:
            print("Error:", e)
        return decoded_labels.copy()



def get_files (path, extention):
    
    files = []
    for root, dirNames, fileNames in os.walk(path):
        if 'All' in root : continue
        for fileName in fnmatch.filter(fileNames, '*' + extention):
            files.append(os.path.join(root, fileName))
    return files



def load_data (path, extention = '.wav'):

    if not os.path.isdir(path):
        print('Dataset path does not exist ')
        quit()
    else:
        # Retrieve all files in chosen path with the specific extension
        file_paths = get_files(path, extention)
        # Get the directory name as its label
        labels = [os.path.basename(os.path.dirname(path)) for path in file_paths]
        if len(file_paths) == 0:
            print('There is no sample in dataset')
            quit()
        else:
            le = LaberEncoder(labels, order = np.array(sorted(np.unique(labels), key = lambda x: (not x.startswith('CN'), x))))
            labels = le.encode(labels)
            df = pd.DataFrame({"path": file_paths, "label": labels})
            df['key'] = df['path'].str.extract('(\w+).wav')

            print(le.map)
            return df, le
        


def segment_audios (audio_paths, output_dir, segment_length = 20, min_acceptable = 10, overlap = 0, target_sr = 16000):
    """
    This function will split the audio files into (n-seconds) segments and save
    each segment as a separate audio file in the specified path ('output_dir')
    ...
    Parameters:
      min_acceptable: Minimum acceptable length (second) for the last segment
      output_dir: The path where segmented audios stored
      target_sr: Target sampling rate
      overlap: Percentage of overlap between consecutive segments.
    """
    for audio_path in tqdm(audio_paths, desc = "Segmenting audio files"):
        audio, sr = torchaudio.load(audio_path)

        resampler = transforms.Resample(orig_freq = sr, new_freq = target_sr)
        # Apply the resampling transform to the audio waveform
        audio = resampler(audio)

        segment_samples = segment_length * target_sr
        overlap_samples = int(segment_samples * overlap)
        step_samples = segment_samples - overlap_samples
        num_segments = (int(audio.size(1)) - segment_samples) // step_samples + 1

        segments = []
        end_sample = 0
        for i in range(num_segments):
            start_sample = i * step_samples
            end_sample = start_sample + segment_samples
            segment = audio[:, start_sample:end_sample]
            segments.append(segment)

        remaining_part = audio[:, end_sample:]
        # If the length of the remaining part is more than 'min_acceptable' seconds,
        # add it as the last segment.
        if remaining_part.size(1) >= min_acceptable * target_sr:
            segments.append(remaining_part)

        audio_name = os.path.basename(audio_path)
        # For each sample (audio file), the parent directory name serves as its label
        label = os.path.basename(os.path.dirname(audio_path))

        # Create destination directory if it not exist.
        des_path = output_dir + '/' + label + '/'
        os.makedirs(des_path, exist_ok = True)

        for i, seg in enumerate(segments):
            output_path = os.path.join(des_path, audio_name + f"segment_{i}.wav")
            torchaudio.save(output_path, seg, target_sr)

