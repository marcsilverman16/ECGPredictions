#!/usr/bin/env python3
"""Segment ECGs by R peaks and write to disk.

The main function is get_r_peak_info.
"""

import emrich_vg.src.vg_beat_detectors as vg
from utilities import read_matrix, get_wanted_file_paths

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

def main():
    user_args = get_user_args()
    hf_info_path = Path(user_args.inp)
    r_peak_info_dir = Path(user_args.out)
    lead_II_parent_dir = Path(user_args.pl2)
    sampling_freq = int(user_args.rate)
  
    r_peak_info_dir.mkdir(parents=True, exist_ok=True)
    r_peak_info_path = Path(r_peak_info_dir, "r_peak_info.parquet.snappy")

    r_peak_info = get_r_peak_info(
        hf_info_path=hf_info_path,
        lead_II_parent_dir=lead_II_parent_dir,
        sampling_freq=sampling_freq
    )

    (r_peak_info
        .collect()
        .write_parquet(
            file=r_peak_info_path,
            compression="snappy"
        )
    )
    return None

def get_r_peak_info(
    hf_info_path:Path|str,
    lead_II_parent_dir:Path|str,
    sampling_freq:int
) -> pl.LazyFrame:
    # Handle args
    hf_info_path = Path(hf_info_path)
    lead_II_parent_dir = Path(lead_II_parent_dir)

    # Main logic
    # Get the info. on which patient_ids we want to 
    # find the R peaks for in their lead II files.
    hf_info = pl.scan_parquet(source=hf_info_path)

    # For the patients that we want to analyze,
    # where are their pre-processed lead II files?
    patient_ids_sorted_wanted = (hf_info
        .select(pl.col("patient_id"))
        .sort(by="patient_id")
        .collect()
        .to_series()
        .to_list()
    )

    lead_II_paths_sorted = get_wanted_file_paths(
        parent_dir=lead_II_parent_dir,
        file_stems=patient_ids_sorted_wanted
    )

    # Unfortunately, some of our patients may not have
    # a lead II data file.
    patient_ids_sorted_actual = [p.stem for p in lead_II_paths_sorted]

    r_peak_info = pl.LazyFrame(
        data=pl.Series(
            name="patient_id", 
            values=patient_ids_sorted_actual,
            dtype=pl.Utf8
        )
    )
    # Read ECGs one at a time from the paths in lead_II_paths_sorted.
    # For each ECG:
    #
    #   Detect R peaks.
    #   Save the R peaks in r_peak_info for the current patient.
    #
    # Write r_peak_info to disk.

    r_peak_detector = vg.FastNVG(
        sampling_frequency=sampling_freq
    )
   
    # r_peaks_for_all_patients will contain the indices
    # of the r_peaks along the time axis.  
    r_peaks_for_all_patients = pl.Series(name="r_peaks", dtype=pl.List(pl.Int64))
    total_to_analyze = len(lead_II_paths_sorted)
    for j, ecg_path in enumerate(lead_II_paths_sorted):
        ecg = read_matrix(ecg_path)
        lead_II = ecg.ravel()
        
        r_peaks = r_peak_detector.find_peaks(sig=lead_II)
        r_peaks_series = pl.Series(values=[r_peaks], dtype=pl.List(pl.Int64))
  
        # Save the r_peaks that we found for the current patient.
        r_peaks_for_all_patients.append(r_peaks_series)
        
        # print("r_peaks_for_all_patients")
        # print(r_peaks_for_all_patients)
        # print("pl.DataFrame(r_peaks_for_all_patients)")
        # print(pl.DataFrame(r_peaks_for_all_patients))
        # if j == 3:
        #     break
        if (j+1) % 1000 == 0:
            print(f"Finished processing {j+1} out of {total_to_analyze} files.")
    
    r_peak_info = (r_peak_info
        .with_columns([
            r_peaks_for_all_patients
            # rechunk to help with memory optimization
            .rechunk(in_place=True)
            .alias("r_peak")
        ])           
    )    

    return r_peak_info


def get_user_args() -> argparse.Namespace:
    """Get arguments from the command line."""
    parser = argparse.ArgumentParser()
    # Optional arguments are prefixed by single or double dashes.
    # The remaining arguments are positional.
    parser.add_argument(
        "--inp", 
        required=True,
        help="Path to a .parquet file of containing column headers \
of 'patient_id', 'age', 'sex', 'snomed_concept_name' and 'ecg_path'. \
Each row should be for a unique patient."
    )

    parser.add_argument(
        "--out", 
        required=True,
        help="Path to a directory (it will be created if it does \
not exist) for the output to be saved within."
    )

    parser.add_argument(
        "--pl2", 
        required=True,
        help="Path to a directory containing MATLAB files of pre-processed \
lead 2 matrices."
    )

    parser.add_argument(
        "--rate", 
        required=False,
        default=500,
        help="The sampling rate for the electrocardiogram (in Hz). The \
default is 500 Hz."
    )
   

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    sys.exit(main())