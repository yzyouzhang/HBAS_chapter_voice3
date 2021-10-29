#!/usr/bin/python3

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import librosa
from torch.utils.data.dataloader import default_collate
import torchaudio


class ASVspoof2019LA(Dataset):
    def __init__(self, path_to_audio='/data/neil/DS_10283_3336/', path_to_features='/data2/neil/ASVspoof2019LA/',
                 part='train', feature='LFCC', feat_len=750, genuine_only=False):
        super(ASVspoof2019LA, self).__init__()
        self.path_to_audio = path_to_audio
        self.path_to_features = path_to_features
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        self.feat_len = feat_len
        self.feature = feature
        self.genuine_only = genuine_only
        self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                      "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                      "A19": 19}
        self.label = {"spoof": 1, "bonafide": 0}
        self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")
        if self.genuine_only:
            if self.part in ["train", "dev"]:
                num_bonafide = {"train": 2580, "dev": 2548}
                self.all_files = self.all_files[:num_bonafide[self.part]]
            else:
                res = []
                for item in self.all_files:
                    if "bonafide" in item:
                        res.append(item)
                self.all_files = res
                assert len(self.all_files) == 7355

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        basename = os.path.basename(filepath)
        all_info = basename.split(".")[0].split("_")
        filename = "_".join(all_info[1:4])
        if self.feature != "Raw":
            featureTensor = torch.load(filepath)
            this_feat_len = featureTensor.shape[1]
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len, :]
            elif this_feat_len < self.feat_len:
                featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
        else:
            # file_path = os.path.join(self.path_to_audio, "LA/ASVspoof2019_LA_" + self.part, "flac", filename+".flac")
            # featureTensor, sr = torchaudio.load(file_path)
            featureTensor = torch.load(filepath)
            this_feat_len = featureTensor.shape[1]
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len]
            if this_feat_len < self.feat_len:
                featureTensor = repeat_padding_RawTensor(featureTensor, self.feat_len)

        tag = self.tag[all_info[4]]
        label = self.label[all_info[5]]
        return featureTensor, filename, tag, label, 2019

    def collate_fn(self, samples):
        return default_collate(samples)


class VCC2020(Dataset):
    def __init__(self, path_to_features="/data2/neil/VCC2020/", feature='LFCC',
                 feat_len=750, genuine_only=False):
        super(VCC2020, self).__init__()
        self.ptf = path_to_features
        self.feat_len = feat_len
        self.feature = feature
        self.tag = {"-": 0, "SOU": 20, "T01": 21, "T02": 22, "T03": 23, "T04": 24, "T05": 25, "T06": 26, "T07": 27, "T08": 28, "T09": 29,
                    "T10": 30, "T11": 31, "T12": 32, "T13": 33, "T14": 34, "T15": 35, "T16": 36, "T17": 37, "T18": 38, "T19": 39,
                    "T20": 40, "T21": 41, "T22": 42, "T23": 43, "T24": 44, "T25": 45, "T26": 46, "T27": 47, "T28": 48, "T29": 49,
                    "T30": 50, "T31": 51, "T32": 52, "T33": 53, "TAR": 54}
        self.label = {"spoof": 1, "bonafide": 0}
        self.genuine_only = genuine_only
        self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")

    def __len__(self):
        if self.genuine_only:
            return 220
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        basename = os.path.basename(filepath)
        all_info = basename.split(".")[0].split("_")
        featureTensor = torch.load(filepath)
        this_feat_len = featureTensor.shape[1]
        if this_feat_len > self.feat_len:
            startp = np.random.randint(this_feat_len - self.feat_len)
            featureTensor = featureTensor[:, startp:startp + self.feat_len, :]
        if this_feat_len < self.feat_len:
            featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
        tag = self.tag[all_info[-2]]
        label = self.label[all_info[-1]]
        return featureTensor, basename, tag, label, 2020

    def collate_fn(self, samples):
        return default_collate(samples)


class ASVspoof2015(Dataset):
    def __init__(self, path_to_features, part='train', feature='LFCC', feat_len=750,
                 genuine_only=False):
        super(ASVspoof2015, self).__init__()
        self.path_to_features = path_to_features
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        self.feat_len = feat_len
        self.feature = feature
        self.tag = {"human": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5,
                    "S6": 6, "S7": 7, "S8": 8, "S9": 9, "S10": 10}
        self.label = {"spoof": 1, "human": 0}
        self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        basename = os.path.basename(filepath)
        all_info = basename.split(".")[0].split("_")
        assert len(all_info) == 4
        featureTensor = torch.load(filepath)
        this_feat_len = featureTensor.shape[1]
        if this_feat_len > self.feat_len:
            startp = np.random.randint(this_feat_len - self.feat_len)
            featureTensor = featureTensor[:, startp:startp + self.feat_len, :]
        if this_feat_len < self.feat_len:
            featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
        filename =  all_info[1]
        tag = self.tag[all_info[-2]]
        label = self.label[all_info[-1]]
        return featureTensor, filename, tag, label, 2015

    def collate_fn(self, samples):
        return default_collate(samples)


class ASVspoof2019LA_DeviceAdversarial(Dataset):
    def __init__(self, path_to_features="/data2/neil/ASVspoof2019LA/", path_to_deviced="/dataNVME/neil/ASVspoof2019LADevice",
                 part="train", feature='LFCC', feat_len=750):
        super(ASVspoof2019LA_DeviceAdversarial, self).__init__()
        self.path_to_features = path_to_features
        suffix = {"train" : "", "dev":"Dev", "eval": "Eval"}
        self.path_to_deviced = path_to_deviced + suffix[part]
        self.path_to_features = path_to_features
        self.ptf = os.path.join(path_to_features, part)
        self.feat_len = feat_len
        self.feature = feature
        self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                    "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                    "A19": 19}
        self.label = {"spoof": 1, "bonafide": 0}
        self.devices = ['AKSPKRS80sUk002-16000', 'AKSPKRSVinUk002-16000', 'Doremi-16000', 'RCAPB90-16000',
                        'ResloRBRedLabel-16000', 'AKSPKRSSpeaker002-16000', 'BehritoneirRecording-16000',
                        'OktavaML19-16000', 'ResloRB250-16000', 'SonyC37Fet-16000']
        if part == "eval":
            self.devices = ['AKSPKRS80sUk002-16000', 'AKSPKRSVinUk002-16000', 'Doremi-16000', 'RCAPB90-16000',
                            'ResloRBRedLabel-16000', 'AKSPKRSSpeaker002-16000', 'BehritoneirRecording-16000',
                            'OktavaML19-16000', 'ResloRB250-16000', 'SonyC37Fet-16000',
                            'iPadirRecording-16000', 'iPhoneirRecording-16000']
        self.original_all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")
        self.deviced_all_files = [librosa.util.find_files(os.path.join(self.path_to_deviced, devicex), ext="pt") for devicex in self.devices]

    def __len__(self):
        return len(self.original_all_files) * (len(self.devices) + 1)

    def __getitem__(self, idx):
        device_idx = idx % (len(self.devices) + 1)
        filename_idx = idx // (len(self.devices) + 1)
        if device_idx == 0:
            filepath = self.original_all_files[filename_idx]
        else:
            filepath = self.deviced_all_files[device_idx-1][filename_idx]
        basename = os.path.basename(filepath)
        all_info = basename.split(".")[0].split("_")
        featureTensor = torch.load(filepath)
        this_feat_len = featureTensor.shape[1]

        if this_feat_len > self.feat_len:
            startp = np.random.randint(this_feat_len - self.feat_len)
            featureTensor = featureTensor[:, startp:startp + self.feat_len, :]
        if this_feat_len < self.feat_len:
            featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
        filename = "_".join(all_info[1:4])
        tag = self.tag[all_info[4]]
        label = self.label[all_info[5]]
        return featureTensor, filename, tag, label, device_idx

    def collate_fn(self, samples):
        return default_collate(samples)


def repeat_padding_Tensor(spec, ref_len):
    mul = int(np.ceil(ref_len / spec.shape[1]))
    spec = spec.repeat(1, mul, 1)[:, :ref_len, :]
    return spec

def repeat_padding_RawTensor(raw, ref_len):
    mul = int(np.ceil(ref_len / raw.shape[1]))
    raw = raw.repeat(1, mul)[:, :ref_len]
    return raw

