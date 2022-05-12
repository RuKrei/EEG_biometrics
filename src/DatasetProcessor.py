import mne
import glob
import os
import pandas as pd


class SplitAndSaveEEG:
    def __init__(self, split_length, data_out) -> None:
        self.split_length = split_length
        self.data_out = data_out

    def split_EEG(self, raw, target) -> None:
        raw.load_data()
        target_stem = os.path.join(self.data_out, target)
        seconds = len(raw.get_data().T) / raw.info["sfreq"]
        n_rawfiles = int(seconds / self.split_length) - 1
        start = 0
        stop = start + self.split_length
        existing_files = glob.glob(os.path.join(self.data_out, target + "*"))
        print(f"The following files exist from this subject:\n{existing_files}\n\n")
        print("The following files have been created for this subject:\n")
        for i in range(n_rawfiles):
            start += self.split_length
            stop = start + self.split_length
            cropfile = raw.copy().crop(tmin=start, tmax=stop)
            idx = len(existing_files) + i + 1
            filename = target + "_split_" + str(idx) + ".edf"
            mne.export.export_raw(filename, cropfile, fmt="edf")
            print(f"Index = {idx}, Now writing: {filename}\n")


class EEGTransformer:
    def __init__(
        self, desired_channels, target_sfreq, lowpass, highpass, n_jobs
    ) -> None:
        self.desired_channels = desired_channels
        self.target_sfreq = target_sfreq
        self.lowpass = lowpass
        self.highpass = highpass
        self.n_jobs = n_jobs

    def _drop_unwanted_channels(self, raw):
        raw = mne.io.read_raw_edf(raw, preload=True)
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

    def transformEEG(self, raw):
        raw, ch_mapping = self._drop_unwanted_channels(raw)
        raw = self._filter(raw)
        raw = self._resampleEEG(raw)
        return raw, ch_mapping

if __name__ == "__main__":
    pass