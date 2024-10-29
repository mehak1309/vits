#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 input_folder output_folder num_workers sampling_rate"
    exit 1
fi

input_folder="${1%/}"  # Remove trailing '/' if present
output_folder="$2"
num_workers="$3"
sampling_rate="$4"

# Check if the output directory exists; if not, create it
if [ ! -d "$output_folder" ]; then
    mkdir -p "$output_folder"
fi

# Define a function to print the ffmpeg command (without executing)
print_ffmpeg_command() {
    input_file="$1"
    local_output_folder="$2"
    local_input_folder="$3"
    local_sampling_rate="$4"
    output_file="${local_output_folder}/$(basename "${input_file%.*}.wav")"  # Output file path with .wav extension
    ffmpeg -y -i "$input_file" -ar "$local_sampling_rate" -ac 1 "$output_file"
}

export -f print_ffmpeg_command

# Use parallel to print ffmpeg commands fori files in parallel
find "$input_folder" \( -name '*.wav' -o -name '*.fqlac' \) | \
parallel -j"$num_workers" print_ffmpeg_command {} "$output_folder" "$input_folder" "$sampling_rate"


# for i in $1/*.wav; do
#   ffmpeg -y -i $i -ar 24000 -ac 1 $2/${i##*/};
# done

# ./resample_audios.sh /home/tts/ttsteam/datasets/google_crowdsourced /home/tts/ttsteam/datasets/google_crowdsourced/wavs-22k 128 22000

# ./resample_audios.sh /home/tts/ttsteam/datasets/indictts/hi/wavs-22k /home/tts/ttsteam/datasets/indictts/hi/wavs-24k 128 24000
# ./resample_audios.sh /home/tts/ttsteam/datasets/wavs_24K /home/tts/ttsteam/datasets/ai4b_internal/rasa/wavs-22k 128 22000
# for i in *.wav;   do name=`echo "$i" | cut -d'.' -f1`;   echo "$name";   ffmpeg -i "$i" -ar 24000 -ac 1 "../wavs-22k/${name}.wav"; done


# ./resample_audios.sh /home/tts/ttsteam/repos/VoiceCraft/iclr_oov_synthgen/indictts/hi/f/samples/ /home/tts/ttsteam/datasets/vc_synth_oov/hi_samples/wavs-24k 128 24000
