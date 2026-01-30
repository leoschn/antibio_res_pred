import glob
import os
import pickle
import pickle as pkl
import numpy as np
from matplotlib import image as mpimg
from pyRawMSDataReader.pyRawMSDataReader.WiffFileReader_py import WiffFileReader

import re

def transform(name):
    m = re.match(r'([A-Z]+)-(\d+)-([A-Z]+)', name)
    if m:
        return f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
    return None

def build_image_ms1_wiff(path, bin_mz):
    #load raw data
    rawFile = WiffFileReader(path)
    max_cycle=0
    total_scan_number = rawFile.GetLastSpectrumNumber()
    for scanNumber in range (total_scan_number):
        if rawFile.GetMSOrderForScanNum(scanNumber) == 1 :
            ms1_start_mz = rawFile.source.ScanInfos[scanNumber].LowMz
            ms1_end_mz = rawFile.source.ScanInfos[scanNumber].HighMz
            max_cycle+=1

    print('start', ms1_start_mz, 'end', ms1_end_mz)
    total_ms1_mz = ms1_end_mz - ms1_start_mz

    n_bin_ms1 = int(total_ms1_mz // bin_mz)
    size_bin_ms1 = total_ms1_mz / n_bin_ms1
    im = np.zeros([max_cycle, n_bin_ms1])

    cycle = 0
    for scanNumber in range(total_scan_number):
        if rawFile.GetMSOrderForScanNum(scanNumber) == 1:
            masses, intensities = rawFile.GetCentroidMassListFromScanNum(scanNumber)
            line = np.zeros(n_bin_ms1)
            if len(masses) > 0:
                for k in range(len(masses)):
                    line[int((masses[k] - ms1_start_mz) // size_bin_ms1)] += intensities[k]
            im[cycle, :] = line
            cycle += 1

    return im


def build_image_ms2_mzml(path_mzml, out_path=None, bin_mz=1.0):


    # ----------------------------
    # 1. Load mzML
    # ----------------------------
    run = pymzml.run.Reader(path_mzml)

    # ----------------------------
    # 2. Identify MS1 range and DIA windows
    # ----------------------------
    dia_window_set = set()
    max_cycle = 0
    ms1_start_mz = None
    ms1_end_mz = None

    for spec in run:
        if spec.ms_level == 1:
            mzs = spec.mz
            if mzs is not None and len(mzs) > 0:
                if ms1_start_mz is None or min(mzs) < ms1_start_mz:
                    ms1_start_mz = min(mzs)
                if ms1_end_mz is None or max(mzs) > ms1_end_mz:
                    ms1_end_mz = max(mzs)
            max_cycle += 1
        elif spec.ms_level == 2:
            # get isolation window from mzML
            target = spec['isolation window target m/z']
            lo = spec['isolation window lower offset']
            hi = spec['isolation window upper offset']
            dia_window_set.add((target - lo, target + hi))

    dia_windows = sorted(dia_window_set, key=lambda x: (x[0], x[1]))
    window_to_index = {win: i + 1 for i, win in enumerate(dia_windows)}  # MS1 = 0

    start_mz = ms1_start_mz
    end_mz = ms1_end_mz
    total_mz = end_mz - start_mz
    n_bin = int(total_mz // bin_mz)
    size_bin = total_mz / n_bin

    # ----------------------------
    # 3. Allocate images
    # ----------------------------
    list_img = [
        np.zeros((max_cycle, n_bin + 1), dtype=np.float32)
        for _ in range(len(dia_windows) + 1)
    ]

    # ----------------------------
    # 4. Fill images
    # ----------------------------
    cycle = -1  # incremented on MS1

    for spec in run:
        mzs = spec.mz
        ints = spec.i
        if mzs is None or len(mzs) == 0:
            continue

        if spec.ms_level == 1:
            cycle += 1
        if cycle < 0:
            continue

        # binning
        masses = np.array(mzs)
        intensities = np.array(ints)
        bins = ((masses - start_mz) / size_bin).astype(int)
        valid = (bins >= 0) & (bins <= n_bin)
        line = np.zeros(n_bin + 1, dtype=np.float32)
        np.add.at(line, bins[valid], intensities[valid])
        line = np.log10(line + 1)  # safe log

        if spec.ms_level == 1:
            list_img[0][cycle, :] = line
        else:
            target = spec['isolation window target m/z']
            lo = spec['isolation window lower offset']
            hi = spec['isolation window upper offset']
            key = (target - lo, target + hi)
            ind = window_to_index[key]
            list_img[ind][cycle, :] = line

    # ----------------------------
    # 5. Metadata
    # ----------------------------
    # get first and last MS1 scans
    start_rt = None
    end_rt = None
    for spec in run:
        if spec.ms_level == 1:
            rt = spec.scan_time[0]  # get numeric RT
            if start_rt is None:
                start_rt = rt
            end_rt = rt


    meta_data = {
        'n_bin': n_bin,
        'size_bin': size_bin,
        'start_mz': start_mz,
        'end_mz': end_mz,
        'max_cycle': max_cycle,
        'dia_windows': window_to_index,
        'span_rt': end_rt - start_rt,
        'start_rt': start_rt,
        'end_rt': end_rt,
    }

    data_out = {'image': list_img, 'metadata': meta_data}

    if out_path is not None:
        with open(out_path, 'wb') as f:
            print('saving images to', out_path)
            pickle.dump(data_out, f)

    return data_out


def build_dataset_ms2(dir_path):
    for filename in glob.glob(os.path.join(dir_path,'*.mzML')):
        print(filename)


        name = transform(os.path.basename(filename))
        os.makedirs('img_ms1',exist_ok=True)
        save_name = os.path.join('img_ms1', name+'.pkl')
        if not os.path.exists(save_name):
            img = build_image_ms2_mzml(filename, 1)
            with open(save_name,'wb') as file:
                pkl.dump(img,file)


def build_dataset_ms1(dir_path):
    for filename in glob.glob(os.path.join(dir_path,'*.wiff')):
        print(filename)


        name = transform(os.path.basename(filename))
        os.makedirs('img_ms1',exist_ok=True)
        save_name = os.path.join('img_ms1', name+'.pkl')
        if not os.path.exists(save_name):
            img = build_image_ms1_wiff(filename, 1)
            with open(save_name,'wb') as file:
                pkl.dump(img,file)

if __name__ == '__main__':
    # build_dataset_ms1('wiff_data')
    data = build_image_ms2_mzml('wiff_data/CITAMA-5-AER-d200.wiff',bin_mz=1)

