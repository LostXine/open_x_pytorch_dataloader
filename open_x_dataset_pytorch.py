import tensorflow_datasets as tfds
from PIL import Image
import io
import re

import torch
from torchdata.datapipes.iter import FileLister, FileOpener
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


def parse_episode_using_meta(episode: dict, meta: tfds.core.features.FeaturesDict)-> dict:
    def fetch_key_in_meta(key: str) \
        -> tfds.core.features.Tensor | tfds.core.features.Text | tfds.core.features.Image:
        iter_keys = key.split('/')
        key_meta = meta
        for key in iter_keys:
            key_meta = key_meta[key]
        return key_meta
    
    transform = ToTensor() # H, W, C -> C, H, W; 0-255 -> 0-1.0
    
    for key, value in episode.items():
        key_meta = fetch_key_in_meta(key)
        # print(key, key_meta, type(value))
        if isinstance(value, torch.Tensor):
            episode[key] = value.reshape(-1, *key_meta.shape)
        else:
            # another type should be
            # google.protobuf.pyext._message.RepeatedScalarContainer
            if isinstance(key_meta, tfds.core.features.Text):
                # decode text bytes (bytes -> str)
                episode[key] = [raw_bytes.decode('utf8') for raw_bytes in value]
            elif isinstance(key_meta, tfds.core.features.Image):
                # decode image bytes (PNG/JPG -> tensor)
                image_tensors = [transform(Image.open(io.BytesIO(raw_bytes))) for raw_bytes in value]
                image_tensors = torch.stack(image_tensors, dim=0) # B, C, H, W
                episode[key] = image_tensors
            else:
                raise ValueError(f'Unknown data type of {key} - {type(key_meta)}')
    return episode


# Caution: this class will load the whole dataset to your RAM!
# For a huge dataset, please check IterateDataset instead.
class OpenXDataset(Dataset):
    def __init__(self, 
                 tf_dir: str,
                 fetch_pattern: str=r'steps*',
                 sample_length: int=2) -> None:
        """
        Args:
            tf_dir (str): full directory containing the downloaded dataset, including the version number.
            fetch_pattern (str, optional): regular expression utilized to specify the data you wish to retrieve. Defaults to r'steps*'.
            sample_length (int, optional): number of transitions per sample. Defaults to 2.
        """
        super().__init__()
        
        # fetch all the keys that match this regular expression
        self.fetch_pattern = fetch_pattern
        
        assert sample_length > 0, \
            'The number of transitions in each sample must be larger than 0'
        self.sample_length = sample_length
        
        # Load metadata using tfds
        b = tfds.builder_from_directory(builder_dir=tf_dir)
        self.info = b.info
        self.meta = b.info.features
        
        # Load data using torch
        datapipe1 = FileLister(tf_dir, "*.tfrecord*")
        datapipe2 = FileOpener(datapipe1, mode="b")
        tfrecord_loader_dp = datapipe2.load_from_tfrecord()
        
        # container for all the data
        self.episodes: list[dict] = []
        # sample index -> (index of the episode, index within the episode)
        self.episode_idx: list[tuple(int, int)] = [] 
        
        episode_start = 0
        for i, episode in enumerate(tfrecord_loader_dp):
            # TODO: filter out invalid episodes using metadata
            self.episodes.append(
                parse_episode_using_meta(episode, self.meta)
                )
            # count the total number of samples we can extract from an episode
            episode_length = episode['steps/is_first'].shape[0]
            samples_per_episode = episode_length - sample_length + 1
            episode_start += samples_per_episode
            self.episode_idx.extend([(i, j) for j in range(episode_start)])

        assert len(self.episodes), 'Fail to load any episodes'
        self.episode_keys = list(self.episodes[0].keys())
    
    def __repr__(self) -> str:
        result = self.info.__repr__()
        result += '\n' + '=' * 10
        result += f'\nTotal episodes: {len(self.episodes)}'
        result += f'\nTotal samples: {self.__len__()} given sample length: {self.sample_length}'
        result += '\n' + '=' * 10
        
        output_keys: list[str] = []
        masked_keys: list[str] = []
        for key in self.episodes[0]:
            if re.search(self.fetch_pattern, key):
                output_keys.append(key)
            else:
                masked_keys.append(key)
        
        result += f'\nOutput keys:\n - ' + '\n - '.join(output_keys)
        result += f'\nMasked keys:\n - ' + '\n - '.join(masked_keys)
        return result
    
    def __len__(self) -> int:
        return len(self.episode_idx)
    
    def __getitem__(self, index: int) -> dict:
        episode_idx, transition_idx = self.episode_idx[index]
        return {
            key: value[transition_idx: transition_idx + self.sample_length]
            for key, value in self.episodes[episode_idx].items()
            if re.search(self.fetch_pattern, key)
        }
    
    
if __name__ == '__main__':
    d = OpenXDataset(
        tf_dir='datasets/asu_table_top_converted_externally_to_rlds/0.1.0/',
        fetch_pattern=r'.*image.*',
        sample_length=8,
    )
    print(d)
    for k, v in d[0].items():
        print(k, v.shape if isinstance(v, torch.Tensor) else len(v))
    