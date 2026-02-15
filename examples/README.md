# 🎤 Lyrics Transcription

Download checkpoint using any of the following command:
```
hf download --local-dir './ckpt/HeartTranscriptor-oss' 'HeartMuLa/HeartTranscriptor-oss' 
modelscope download --model 'HeartMuLa/HeartTranscriptor-oss' --local_dir './ckpt/HeartTranscriptor-oss'
```

```
python ./examples/run_lyrics_transcription.py --model_path=./ckpt
```

By default this command will load the generated music file at `./assets/output.mp3` and print the transcribed lyrics. Use `--music_path` to specify the path to the music file.

Note that our HeartTranscriptor is trained on separated vocal tracks. In this example usage part, we directly demonstrate on unseparated music tracks, which is purely for simplicity of illustration. We recommend using source separation tools like demucs to separate the tracks before transcribing lyrics to achieve better results.
