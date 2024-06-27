
# VITS Model Training

This README outlines the steps to train a VITS (Variational Inference for Text-to-Speech) model. Follow the instructions below to set up your environment, prepare your data, and run the training scripts in sequence.

**Note:** Use three-letter Unicode naming for Indian languages like `mar` (Marathi), `hin` (Hindi), and `guj` (Gujarati), etc. Replace `<lang>` with the appropriate language code.

## Steps

### A. Data Preprocessing

&nbsp;1. **Download the Dataset from Graphana:**
* Go to the following link: Graphana TTS Dashboard
* Choose the language, gender, status, style, and category.

    | Settings   | Value  |
    | ------------- | ------------- |
    |  status  | accepted |
    | gender  | male/female or both |
    |  style  | required style |
    | category  | all |

* Download the CSV file.

&nbsp;2. **Download Audio Files:**

 - The CSV file contains links to audio files. Download all audio files.
- Transfer all audio files along with the CSV file to the CDAC server.

- Save the audio files and the CSV file in: `
    ~/ttsteam/datasets/ai4bharat_internal/<lang>/
    `
  
&nbsp;3. **Set Up CDAC Environment:**

- Create a screen session and enter a tmux session:
    ```bash
    screen -rd <name>
    tmux new -s <session_name>
    ```

- Deactivate existing environments and activate the required environment:
    ```bash
    conda deactivate
    conda activate walnut_vits
    ```

&nbsp;4. **Extract the Audio:**

- Run the Python script located at: `~/ttsteam/repos/vits/text/extract_file.py`

- It saves the audio files to the following path: `~/ttsteam/datasets/pilot_rasa/audio_48k/<lang>`

&nbsp;5. **Sample the Audio:**

- The script is located at: `~/ttsteam/repos/audio_scripts/resample_audios.sh`

- Run the following command to resample the audio files, using a target sampling rate of 24,000 Hz:

    ```bash
    ~/ttsteam/repos/audio_scripts/resample_audios.sh <input_folder> <output_folder> <num_worker_threads> <target_sampling_rate>
    ```

- It saves the audio files to the following path: `~/ttsteam/datasets/pilot_rasa/audio/<lang>`

&nbsp;6. **Modify the CSV File:**

- Edit the following variables in the script:
    ```bash
    input_csv_path = '~/ttsteam/datasets/pilot_rasa/csv_file/<transferred_csv_file_name>'
    output_directory = '~/ttsteam/datasets/indictts/<lang>/'
    audio_path_prefix = '~/ttsteam/datasets/pilot_rasa/audio/<lang>'
    ```

- Run the script to modify the CSV file for model training:
    ```bash
    python ~/ttsteam/repos/vits/text/modify_csv.py
    ```

    This script splits the CSV file into train and test CSV files and saves them in the following location:
    - Train file: `~/ttsteam/datasets/indictts/<lang>/metadata_train_vits_raw.csv`
    - Test file: `~/ttsteam/datasets/indictts/<lang>/metadata_test_vits_raw.csv`

&nbsp;7. **Create JSON  file:**

* A JSON file needs to be created at the following filepath: `~/ttsteam/repos/vits/configs/pilot_configs/indictts_<lang>_raw.json`


* Modify the values of the following variables:



    | Variable   | Value  |
    | ------------- | ------------- |
    | sampling_rate | 24000 |
    | n_speakers  | <integer_value> |
    | n_emotions  | <integer_value> |
    | batch_size  | <integer_value> |
  


* Change the file path where you have stored the json file for these 2 variables:

    training_files: `~/ttsteam/datasets/indictts/<lang>/metadata_train_vits_raw.csv`

    validation_files: `~/ttsteam/datasets/indictts/<lang>/metadata_test_vits_raw.csv`


&nbsp;8. **Add Symbols (Step not required):**

- Go to `~/ttsteam/repos/vits/text/symbols.py`.
- Change the `_letters_all` variable to include characters of all languages and punctuation marks.
      
&nbsp; &nbsp;**Note:** This step has already been done and can be skipped.

### B. Model Training
    
 &nbsp;1. **Train the Model:**


```bash
setproxy
cd ~ttsteam/repos/vits
python train_ms.py -c configs/pilot_configs/indictts_<lang>_raw.json -m pilot_rasa/indictts_<lang>
```

The training logs will be saved in this file path: 

```bash 
~/ttsteam/repos/vits/logs/pilot_rasa/indictts_<lang>
```

### C. Inference
    
 &nbsp;1. **Test the model:**

&nbsp;&nbsp;Update the file paths in inference.py for the following variables:

```bash
hps = utils.get_hparams_from_file("/~/ttsteam/repos/vits/configs/pilot_configs/indictts_<lang>_raw.json")
_ = utils.load_checkpoint("/~/ttsteam/repos/vits/logs/pilot_rasa/indictts_<lang>/<last_checkpoint>.pth", net_g, None)
df = pd.read_csv("/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/<lang>/metadata_test_vits_raw.csv", header=None, sep="|")
audio_path = f'/nlsasfs/home/ai4bharat/praveens/ttsteam/repos/vits/evaluated/pilot_rasa/<lang>/val'
```

Then, execute the following commands to run the inference:

```bash
tmux new -s <session_name>
conda deactivate
conda activate walnut_vits
cd ~ttsteam/repos/vits
python inference.py
```

&nbsp;The audio files will be saved at the following path:

```bash 
~/ttsteam/repos/vits/evaluated/pilot_rasa/<lang>/val
```


## Additional Information

- **Training Tips:**
    - Ensure your dataset is normalized and well-preprocessed.
    - Monitor the training process and adjust hyperparameters as necessary.

- **Troubleshooting:**
    - Check error logs and ensure all dependencies are installed correctly.
    - Refer to the official VITS repository and documentation for additional help and resources.


## Meta
Distributed under the MIT license. See ``LICENSE`` for more information.
