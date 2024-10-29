import pandas as pd
import os
from tqdm import tqdm


emotions_mapping = {
        'SANGRAH': 0,
        'WIKI': 0,
        'INDIC':0,
        'PROPER NOUN':0,
        'HAPPY' : 1,
        'SAD':2,
        'ANGER' :3,
        'FEAR': 4,
        'SURPRISE':5,
        'DISGUST':6,
        'ALEXA':7,
        'DIGI':8,
        'BOOK':9,
        'CONV':10,
        'NEWS':11,
        'UMANG':12,
        'BB':13,
    }

def get_gender_from_code(code):
    if code == 'm':
        return 'male'
    return 'female'

def convert_filename_to_filepath(filename):
    lang_code = filename[:2]
    
    gender_code = filename[2]

    gender = get_gender_from_code(gender_code)

    if lang_code == 'ta' and gender_code == 'g':
        gender = 'male'

    # if lang_code == 'ne':
    #     filepath = f"/home/tts/ttsteam/datasets/google_crowdsourced/{lang_code}/wavs/{filename}"
    # else: 
    filepath = f"/home/tts/ttsteam/datasets/google_crowdsourced/wavs-22k/{filename}"

    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        exit()

    return filepath

def update_dataframe_with_filepaths(df):
    df['filepath'] = df['filename'].apply(convert_filename_to_filepath)
    df.drop(columns=['filename'], inplace=True)
    return df

speakers = 0
# -------------------------------RASA--------------------------------------------------

rasa_folder = '/home/tts/ttsteam/datasets/ai4b_internal'
languages = ['as', 'bn', 'ta']

all_train_data = []
all_test_data = []
for lang in languages:
    wavs_folder = '/home/tts/ttsteam/datasets/ai4b_internal/{lang}/wavs-22k/'
    train_df_1 = pd.read_csv(f'/home/tts/ttsteam/datasets/ai4b_internal/{lang}/metadata/expressives/metadata_train_1hr.csv', sep = '|', header = None, names = ['filepath', 'text', 'speaker', 'lang', 'emotion'])
    train_df_2 = pd.read_csv(f'/home/tts/ttsteam/datasets/ai4b_internal/{lang}/metadata/neutral/metadata_train_10hr.csv', sep = '|', header = None, names = ['filepath', 'text', 'speaker', 'lang', 'emotion'])

    train_df = pd.concat([train_df_1, train_df_2])

    test_df_1 = pd.read_csv(f'/home/tts/ttsteam/datasets/ai4b_internal/{lang}/metadata/expressives/metadata_test_1hr.csv', sep = '|', header = None, names = ['filepath', 'text', 'speaker', 'lang', 'emotion'])
    test_df_2 = pd.read_csv(f'/home/tts/ttsteam/datasets/ai4b_internal/{lang}/metadata/neutral/metadata_test_10hr.csv', sep = '|', header = None, names = ['filepath', 'text', 'speaker', 'lang', 'emotion'])

    test_df = pd.concat([test_df_1, test_df_2])

    train_df['filepath'] = wavs_folder + train_df['filepath'] + '.wav'
    test_df['filepath'] = wavs_folder + test_df['filepath'] + '.wav'
    
    unique_speakers = train_df['speaker'].unique()

    Rasa_spk_map = {speaker: speakers + idx for idx, speaker in enumerate(unique_speakers)}

    speakers += len(unique_speakers)
    
    train_df['speaker'] = train_df['speaker'].map(Rasa_spk_map)
    test_df['speaker'] = test_df['speaker'].map(Rasa_spk_map)
    train_df['emotion'] = train_df['emotion'].map(emotions_mapping)
    test_df['emotion'] = test_df['emotion'].map(emotions_mapping)

    all_train_data.append(train_df)
    all_test_data.append(test_df)

    train_df.to_csv(f'/home/tts/ttsteam/repos/vits/manifests/rasa/{lang}_train.csv', index=False, sep='|', header=None)
    test_df.to_csv(f'/home/tts/ttsteam/repos/vits/manifests/rasa/{lang}_test.csv', index=False, sep='|', header=None)


combined_train_df = pd.concat(all_train_data, ignore_index=True)
combined_test_df = pd.concat(all_test_data, ignore_index=True)

combined_train_df.to_csv('/home/tts/ttsteam/repos/vits/manifests/rasa/train.csv', index=False, sep='|', header=None)
combined_test_df.to_csv('/home/tts/ttsteam/repos/vits/manifests/rasa/test.csv', index=False, sep='|', header=None)

print ('Processed Rasa')


# -------------------------------IndicTTS----------------------------------------------

itts_folder = '/home/tts/ttsteam/datasets/indictts'
languages = ['as', 'bn', 'brx', 'gu', 'hi', 'kn', 'ml', 'mni', 'mr', 'or', 'pa', 'raj', 'ta', 'te']
all_train_data = []
all_test_data = []
os.makedirs('/home/tts/ttsteam/repos/vits/manifests/indictts/', exist_ok = True)
for lang in languages:
    lang_folder = os.path.join(itts_folder, lang)
    train_csv = os.path.join(lang_folder, 'metadata_train.csv')
    test_csv = os.path.join(lang_folder, 'metadata_test.csv')
    wavs_folder = os.path.join(lang_folder, 'wavs-22k')

    train_df = pd.read_csv(train_csv, sep = '|', header = None, names = ['filename', 'text', 'speaker'])
    test_df = pd.read_csv(test_csv, sep = '|', header = None, names = ['filename', 'text', 'speaker'])

    train_df['filename'] = wavs_folder + '/' + train_df['filename'] + '.wav'
    test_df['filename'] = wavs_folder + '/' + test_df['filename'] + '.wav'

    unique_speakers = train_df['speaker'].unique()
    ITTS_spk_map = {speaker: speakers + idx for idx, speaker in enumerate(unique_speakers)}
    speakers += len(unique_speakers)

    train_df['speaker'] = train_df['speaker'].map(ITTS_spk_map)
    test_df['speaker'] = test_df['speaker'].map(ITTS_spk_map)
    train_df['emotion'] = 0
    test_df['emotion'] = 0

    all_train_data.append(train_df)
    all_test_data.append(test_df)


    train_df.to_csv(f'/home/tts/ttsteam/repos/vits/manifests/indictts/{lang}_train.csv', index=False, sep='|', header=None)
    test_df.to_csv(f'/home/tts/ttsteam/repos/vits/manifests/indictts/{lang}_test.csv', index=False, sep='|', header=None)


combined_train_df = pd.concat(all_train_data, ignore_index=True)
combined_test_df = pd.concat(all_test_data, ignore_index=True)

combined_train_df.to_csv('/home/tts/ttsteam/repos/vits/manifests/indictts/train.csv', index=False, sep='|', header=None)
combined_test_df.to_csv('/home/tts/ttsteam/repos/vits/manifests/indictts/test.csv', index=False, sep='|', header=None)


print ('Processed IndicTTS')

# -----------------------------------GTTS-------------------------------------------------------
train_filepath = '/home/tts/ttsteam/datasets/google_crowdsourced/metadata_train_raw.csv'
test_filepath = '/home/tts/ttsteam/datasets/google_crowdsourced/metadata_test_raw.csv'

df_train = pd.read_csv(train_filepath, sep = '|', header = None, names = ['filename', 'text', 'speaker'])
df_test = pd.read_csv(test_filepath, sep = '|', header = None, names = ['filename', 'text', 'speaker'])

df_train = update_dataframe_with_filepaths(df_train)
df_test = update_dataframe_with_filepaths(df_test)

df_all = pd.concat([df_train, df_test])

unqiue_speakers = df_all['speaker'].unique()

GTTS_spk_map = {speaker: speakers + idx for idx, speaker in enumerate(unique_speakers)}

df_train['speaker'] = df_train['speaker'].map(GTTS_spk_map)
df_test['speaker'] = df_test['speaker'].map(GTTS_spk_map)

df_train['emotion'] = 0
df_test['emotion'] = 0

df_train.to_csv('/home/tts/ttsteam/repos/vits/manifests/gtts/train.csv', sep = '|', header = None, index = False)
df_test.to_csv('/home/tts/ttsteam/repos/vits/manifests/gtts/test.csv', sep = '|', header = None, index = False)

speakers += len(unique_speakers)

print ('Processed GTTS')


# -----------------LIMMITS----------------------------------------------------

limmits_folder = '/home/tts/ttsteam/datasets/limmits'

limmits_lang = ['Bengali_F', 'Chhattisgarhi_F', 'Hindi_F', 'Kannada_F', 'Marathi_F', 'Telugu_F', 'Bengali_M', 'Chhattisgarhi_M', 'Hindi_M', 'Kannada_M', 'Marathi_M', 'Telugu_M']

languages = ['Bengali', 'Chhattisgarhi', 'Hindi', 'Kannada', 'Marathi', 'Telugu']


all_train_data = []
all_test_data = []


for lang in languages:
    folderpath_male = os.path.join(limmits_folder, f'{lang}_M')
    folderpath_female = os.path.join(limmits_folder, f'{lang}_F')

    txt_male = os.path.join(folderpath_male, 'txt')
    wav_male = os.path.join(folderpath_male, 'wav')

    txt_female = os.path.join(folderpath_female, 'txt')
    wav_female = os.path.join(folderpath_female, 'wav')

    df = pd.DataFrame(columns = ['filepath', 'text', 'speaker', 'emotion'])

    for val in ['male', 'female']:
        if val == 'male':
            text_folder = txt_male
            wav_folder = wav_male
        else:
            text_folder = txt_female
            wav_folder = wav_female

        for file in tqdm(os.listdir(text_folder)):
            text_file = os.path.join(text_folder, file)

            with open(text_file, 'r', encoding='utf-8') as f:
                line = f.readline().strip()
            
            wav_path = os.path.join('/home/tts/ttsteam/datasets/limmits/wavs-22k', f'{file[:-4]}.wav')
            data = pd.DataFrame([[wav_path, line, speakers + 1, 0]], columns=['filepath', 'text', 'speaker', 'emotion'])
            df = pd.concat([df, data], ignore_index=True)

        speakers+=1

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = df.groupby('speaker').apply(lambda x: x.sample(frac=0.05, random_state=42)).reset_index(drop=True)
    train_df = df.drop(test_df.index).reset_index(drop=True)
    
    all_train_data.append(train_df)
    all_test_data.append(test_df)
    
    train_df.to_csv(f'/home/tts/ttsteam/repos/vits/manifests/limmits/{lang}_train.csv', index=False, sep='|', header=None)
    test_df.to_csv(f'/home/tts/ttsteam/repos/vits/manifests/limmits/{lang}_test.csv', index=False, sep='|', header=None)


combined_train_df = pd.concat(all_train_data, ignore_index=True)
combined_test_df = pd.concat(all_test_data, ignore_index=True)

combined_train_df.to_csv('/home/tts/ttsteam/repos/vits/manifests/limmits/train.csv', index=False, sep='|', header=None)
combined_test_df.to_csv('/home/tts/ttsteam/repos/vits/manifests/limmits/test.csv', index=False, sep='|', header=None)


print ('Processed Limmits')




