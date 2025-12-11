import glob
import os
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

def build_image_ms2_wiff(path, bin_mz):
    # load raw data
    rawFile = WiffFileReader(path)
    max_cycle = 0

    start_rt = rawFile.GetStartTime()
    end_rt = rawFile.GetEndTime()
    span_rt = end_rt - start_rt

    first_scan, last_scan = rawFile.GetFirstSpectrumNumber(), rawFile.GetLastSpectrumNumber()
    list_precursor_mass_center = []
    for scanNumber in range(first_scan, last_scan):
        if rawFile.GetMSOrderForScanNum(scanNumber) == 1:
            ms1_start_mz = rawFile.source.ScanInfos[scanNumber].LowMz
            ms1_end_mz = rawFile.source.ScanInfos[scanNumber].HighMz
            max_cycle += 1
        elif rawFile.GetPrecursorMassForScanNum(scanNumber) not in list_precursor_mass_center:
            list_precursor_mass_center.append(rawFile.GetPrecursorMassForScanNum(scanNumber))

    print('start', ms1_start_mz, 'end', ms1_end_mz)
    total_ms1_mz = ms1_end_mz - ms1_start_mz

    n_bin_ms1 = int(total_ms1_mz // bin_mz)
    size_bin_ms1 = total_ms1_mz / n_bin_ms1
    list_img = [np.zeros([max_cycle, n_bin_ms1 + 1]) for i in range(len(list_precursor_mass_center) + 1)]
    cycle = 0
    dict_int = {}
    ind = 1

    for mass in list_precursor_mass_center:
        dict_int[mass] = ind
        ind += 1

    for scanNumber in range(first_scan, last_scan):
        masses, intensities = rawFile.GetCentroidMassListFromScanNum(scanNumber)
        line = np.zeros(n_bin_ms1 + 1)
        if len(masses) > 0:
            for k in range(len(masses)):
                line[int((masses[k] - ms1_start_mz) // size_bin_ms1)] += intensities[k]
        if rawFile.GetMSOrderForScanNum(scanNumber) == 1:
            list_img[0][cycle, :] = np.maximum(np.log10(line), np.zeros(n_bin_ms1 + 1))
        else:
            ind = dict_int[rawFile.GetPrecursorMassForScanNum(scanNumber)]
            list_img[ind][cycle, :] = np.maximum(np.log10(line), np.zeros(n_bin_ms1 + 1))
            if rawFile.GetPrecursorMassForScanNum(scanNumber) == list_precursor_mass_center[-1]:
                cycle += 1

    meta_data = {'n_bin_ms1': n_bin_ms1, 'size_bin_ms1': size_bin_ms1, 'ms1_start_mz': ms1_start_mz,
                 'ms1_end_mz': ms1_end_mz, 'max_cycle': max_cycle,
                 'list_precursor_mass_center': list_precursor_mass_center,
                 'total_ms1_mz': total_ms1_mz, 'start_rt': start_rt, 'end_rt': end_rt, 'span_rt': span_rt}

    data_out = {'image': list_img, 'metadata': meta_data}

    return data_out

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
    build_dataset_ms1('wiff_data')