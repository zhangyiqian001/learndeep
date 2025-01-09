import librosa
import numpy as np

# FPS = 30
# HOP_LENGTH = 512
# SR = FPS * HOP_LENGTH

class ShortAudioMelProcessor:
    def __init__(self, fps=30, hop_length=512, ):
        self.fps = fps
        self.hop_length = hop_length
        self.sr = fps * hop_length

    def extract(self, fpath):
        # (31512,)
        data, _ = librosa.load(fpath, sr=self.sr)
        # (62,)
        envelope = librosa.onset.onset_strength(y=data, sr=self.sr)  # (seq_len,)
        # (62, 20)
        mfcc = librosa.feature.mfcc(y=data, sr=self.sr, n_mfcc=20).T  # (seq_len, 20)
        # (62, 12)
        chroma = librosa.feature.chroma_cens(
            y=data, sr=self.sr, hop_length=self.hop_length, n_chroma=12
        ).T  # (seq_len, 12)
        # (9,)
        peak_idxs = librosa.onset.onset_detect(
            onset_envelope=envelope.flatten(), sr=self.sr, hop_length=self.hop_length
        )
        # (62,)
        peak_onehot = np.zeros_like(envelope, dtype=np.float32)
        peak_onehot[peak_idxs] = 1.0  # (seq_len,)


        start_bpm = librosa.beat.tempo(y=librosa.load(fpath)[0])[0]

        tempo, beat_idxs = librosa.beat.beat_track(
            onset_envelope=envelope,
            sr=self.sr,
            hop_length=self.hop_length,
            start_bpm=start_bpm,
            tightness=100,
        )
        # (62,)
        beat_onehot = np.zeros_like(envelope, dtype=np.float32)
        beat_onehot[beat_idxs] = 1.0  # (seq_len,)
        # (62, 35)
        audio_feature = np.concatenate(
            [envelope[:, None], mfcc, chroma, peak_onehot[:, None], beat_onehot[:, None]],
            axis=-1,
        )

        # chop to ensure exact shape
        # audio_feature = audio_feature[:5 * self.fps]
        # assert (audio_feature.shape[0] - 5 * self.fps) == 0, f"expected output to be ~5s, but was {audio_feature.shape[0] / self.fps}"

        #np.save(save_path, audio_feature)
        return audio_feature

class LongAudioMelProcessor:
    def __init__(self):
        pass


if __name__ == '__main__':
    ShortAudioMelProcessor().extract("F:\learndeep\src\data\VCTK-Corpus-0.92\wav48_silence_trimmed\p225\p225_001_mic1.flac")