import soundfile as sf 
import os 
import json
import whisper
import tqdm
import librosa
model = whisper.load_model('large')

def makedirs(output_dir):
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    return output_dir
wav_dir = 'processed_audio'
wav_files = [i for i in os.listdir(wav_dir) if i[-4:] == '.wav']
output_dir = 'transcribe'
target_sr = 16000
language = 'Chinese'

for wav_path in tqdm.tqdm(wav_files):
    wav_path = os.path.join(wav_dir, wav_path)
    wav_name = os.path.basename(wav_path).split('.')[0]
    print(wav_path)
    wav, wav_sr = sf.read(wav_path, always_2d=True)
    wav = wav[:, 0]
    if wav_sr != 16000:
        wav = librosa.resample(wav, orig_sr=wav_sr, target_sr=target_sr)
        seg_wav_path = os.path.join(wav_dir, f'{wav_name}_16kHz.wav')
        sf.write(makedirs(seg_wav_path), wav, samplerate=target_sr)
        wav_path=seg_wav_path
    result = model.transcribe(
        wav_path, language=language, word_timestamps=True,
        without_timestamps=False)
    # print(result)
    
    transcribe_path = f"{output_dir}/{os.path.basename(wav_path)[:-4]}.json"
    transcribe_path = makedirs(transcribe_path)
    with open(transcribe_path, 'w') as write_f:
        json.dump(result, write_f, indent=4, ensure_ascii=False)