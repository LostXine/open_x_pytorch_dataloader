# PyTorch DataLoader for Open-X Embodiment Datasets

An unofficial pytorch dataloader for [Open-X Embodiment Datasets](https://robotics-transformer-x.github.io/).

This README will guide you to integrate the Open-X Embodiment Datasets into your PyTorch project. For a native TensorFlow experience, please check [the official repo](https://github.com/google-deepmind/open_x_embodiment).

## Download the datasets

1. Check available datasets and their corresponding metadata in [the dataset spreadsheet](https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/edit#gid=0)
2. Set your preferred download destination `download_dst` in [generate_download_script.py](generate_download_script.py) and confirm the datasets you want to download. By default, the Python script will create a shell script that downloads all 53 datasets, amounting to a total size of approximately 4.5TB.
3. Follow [this guide](https://cloud.google.com/storage/docs/gsutil_install#linux) to setup `gsutil`
4. Generate the shell script and start to download:
    ```
    python3 generate_download_script.py
    chmod +x download.sh
    ./download.sh
    ```

This section was last updated on 1/19/2024.

## Play with the dataloader

1. Install python dependence
    ```
    pip3 install -r requirements.txt
    ```
2. If your machine has enough RAM to hold the whole dataset, you can init the dataset with `class OpenXDataset(Dataset)` in `open_x_dataset_pytorch.py`. A quick example:

    ```
    d = OpenXDataset(
        tf_dir='datasets/asu_table_top_converted_externally_to_rlds/0.1.0/',
        fetch_pattern=r'.*image.*',
        sample_length=8,
    )
    print(d)
    ```

    * `tf_dir`: full directory containing the downloaded dataset, including the version number.
    * `fetch_pattern`: regular expression utilized to specify the data you wish to retrieve. Defaults to `r'steps*'`. The example above only retrieves visual observations.
    * `sample_length`: number of transitions per sample. If set to `2`, the returned sample will be $[s_1, s_2]$.
    
    The last several lines of the output of the code above:
    ```
    ==========
    Total episodes: 110
    Total samples: 1433503
    ==========
    Output keys:
    - steps/observation/image
    Masked keys:
    - steps/observation/state_vel
    - steps/ground_truth_states/bread
    - steps/is_first
    - steps/ground_truth_states/coke
    - steps/ground_truth_states/cube
    - steps/language_embedding
    - steps/is_terminal
    - steps/is_last
    - steps/discount
    - steps/ground_truth_states/EE
    - steps/language_instruction
    - steps/ground_truth_states/pepsi
    - steps/ground_truth_states/milk
    - steps/observation/state
    - steps/goal_object
    - steps/action
    - episode_metadata/file_path
    - steps/ground_truth_states/bottle
    - steps/action_delta
    - steps/action_inst
    - steps/reward
    ```

    `__getitem__()` returns a dictionary where the keys correspond to `fetch_pattern`. The associated value for each key will be either a tensor of size `(sample_length, *original feature shape)`[^1] or a list with `sample_length` elements. 

3. If the machine does not have enough RAM: I'm still working on it, stay tuned!

## TODO
1. Use `IterableDataset` to save RAM
2. Filter out the invalid episodes according to [the dataset format](https://github.com/google-research/rlds?tab=readme-ov-file#dataset-format)

## Acknowledgment

I really appreciate the substantial open-sourcing effort contributed by the creators of this extensive dataset.
Thank [Jinghuan Shang](https://elicassion.github.io/) for valuable discussions.

[^1]: When the feature is an image, the tensor will have a shape of `(sample_length, C, H, W)` instead.
