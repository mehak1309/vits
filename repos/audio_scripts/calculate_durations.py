# -*- coding: utf-8 -*-
"""
Calculates hours (.wav/.mp3) from a directory

Example usage:
    python calculate_durations.py -d ~/ttsteam/datasets/indictts/pa -o durations_indictts_pa.csv
    python calculate_durations.py -d ~/ttsteam/datasets/SAMHAAR/PS -o durations_samhaar_ps.csv -nw 8
"""

from asyncio import as_completed
import os
import ffmpeg
import shutil
import argparse
import pandas as pd
from tqdm import tqdm
from scipy.io.wavfile import read as read_wav

import concurrent.futures

def get_duration(path):
    try:
        return float(ffmpeg.probe(path)['format']['duration'])
    except Exception as e:
        return 0

def run(args):
    filenames, durations = [], []
    for root, dirs, files in tqdm(os.walk(args.dir)):
        for file in tqdm(files):
            if ('.wav' in file) or ('.mp3' in file):
                filename = os.path.join(root, file)
                # sr, data = read_wav(filename)
                durations.append(get_duration(filename))
                filenames.append(filename)
    
    with tqdm(total=len(filenames)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(get_duration, filename): filename for filename in filenames}
            results = {}
            for future in concurrent.futures.as_completed(futures):
                arg = futures[future]
                results[arg] = future.result()
                pbar.update(1)

    filenames = [f for f in results.keys()]
    durations = [d for d in results.values()]

    df = pd.DataFrame({'filename': filenames, 'duration': durations})
    df = df[df['duration'] > 0]
    df.to_csv(args.output_csv, index=False)

    print('-'*20, 'QUICK_STATS', '-'*20)
    print(df.describe())
    print('-'*50)
    print(df['duration'].sum()/60/60, ' hours of data!')
    print(df['duration'].sum()/60, ' mins of data!')
    print('-'*50)

# 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--dir',
        type=str,
        default='data/',
        help='Directory containing audio files.')
    parser.add_argument(
        '-o',
        '--output_csv',
        type=str,
        default='durations.csv',
        help='Output CSV path with duration of each audio file found.')
    parser.add_argument(
        '-nw',
        '--num_workers',
        type=int,
        default=1,
        help='Number of workers used by ThreadPoolExecutor.'
    )
    args = parser.parse_args()
    run(args)