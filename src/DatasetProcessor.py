import mne
import glob
import os
import pandas as pd


class SplitAndSaveEEG:
    def __init__(self, raw, split_lenght) -> None:
        self.raw = raw
        self.split_length = split_lenght

    def split_EEG(raw, split_length):
        raw = mne.io.read_raw_edf(raw, preload=True)
        if EEG_SOURCE.lower() == "keller":
            print("--> Keller-EEG, renaming channels")
            _prep_keller(raw)
        print(raw.info)
        if not raw.info["sfreq"] == TARGET_SFREQ:
            print(
                f"\n\n\n --> resampling data from {raw.info['sfreq']} to {TARGET_SFREQ} Hz"
            )
            raw.resample(TARGET_SFREQ)
        raw.filter(LOWPASS, HIGHPASS, n_jobs=N_JOBS)

        seconds = len(raw.get_data().T) / raw.info["sfreq"]
        filename_bare = os.path.splitext(eeg_file)[0]

        n_rawfiles = int(seconds / split_length) - 1
        start = 0
        stop = start + split_length

        for i in range(n_rawfiles):
            cropfile = raw.copy().crop(tmin=start, tmax=(stop))
            loop_filename = str(filename_bare + "_split_" + str(i) + ".npy")
            start += split_length
            stop = start + split_length
            np.save(loop_filename, cropfile.get_data())


class EEGTransformer:
    def __init__(
        self, raw, desired_channels, target_sfreq, lowpass, highpass, n_jobs
    ) -> None:
        self.raw = raw
        self.desired_channels = desired_channels
        self.target_sfreq = target_sfreq
        self.lowpass = lowpass
        self.highpass = highpass
        self.n_jobs = n_jobs

    def _drop_unwanted_channels(self):
        raw = mne.io.read_raw_edf(self.raw, preload=True)
        base = raw.info.ch_names
        raw.rename_channels(lambda s: s.strip("EEG "))
        before = raw.info.ch_names
        raw.rename_channels(lambda s: s.split("-REF")[0], allow_duplicates=False)
        raw.rename_channels({"T3": "T7", "T5": "P7", "T4": "T8", "T6": "P8"})
        after = raw.info.ch_names
        ch_mapping = pd.DataFrame(
            {"Before": base, "Strip EEG": before, "Strip -REF": after}
        )
        raw.pick_channels(self.desired_channels)
        return raw, ch_mapping

    def _resampleEEG(self, raw):

        if not raw.info["sfreq"] == self.target_sfreq:
            raw.resample(self.target_sfreq)
        return raw

    def _filter(self, raw):
        raw.filter(self.lowpass, self.highpass, self.n_jobs)
        return raw

    def transformEEG(self):
        raw, ch_mapping = self._drop_unwanted_channels()
        raw = self._filter(raw)
        raw = self._resampleEEG(raw)
        return ch_mapping, raw
