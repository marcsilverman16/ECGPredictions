#!/usr/bin/env python3
"""Minimally preprocess ECG files.  
1. Read in each MATLAB file and select lead II.
2. Detrend and filter out potential power-line interferance.
3. Write the file to disk.
"""

from utilities import \
    read_matrix, \
    get_leads 
import sys
import argparse
from pathlib import Path

from neurokit2 import \
    signal_detrend, \
    signal_filter

from scipy.io import savemat
import numpy as np

def main():
    user_args = get_user_args()
    out_dir = Path(user_args.out)
    wfdb_path = Path(user_args.ecgs)
    sampling_freq = int(user_args.rate)
    power_line_freq = float(user_args.pf)

    # Get ready!
    out_dir.mkdir(parents=True, exist_ok=True)
    # https://docs.python.org/3/library/pathlib.html#pathlib.Path.glob
    ecg_paths = wfdb_path.glob(
        pattern="*/*/*.mat"
    )

    for ecg_path in ecg_paths:
        # Read in the ECG.
        ecg = read_matrix(matlab_file_path=ecg_path)
        lead_II = get_leads("II", ecg=ecg)
        
        # Preprocess lead_II
        # Detrend
        lead_II_detrended = signal_detrend(
            signal=lead_II.ravel(),
            method="tarvainen2002"
            # method="polynomial",
            # order=2
        )

        lead_II_detrended_notch_filtered = signal_filter(
            signal=lead_II_detrended,
            sampling_rate=sampling_freq,
            method="powerline",
            powerline=power_line_freq
        )
        # Write to disk
        out_path = Path(out_dir, ecg_path.name)
        out_dict = {"val": lead_II_detrended_notch_filtered}
        savemat(
            file_name=out_path,
            mdict=out_dict,
        )

    return None
    

def get_user_args() -> argparse.Namespace:
    """Get arguments from the command line."""
    parser = argparse.ArgumentParser()
    # Optional arguments are prefixed by single or double dashes.
    # The remaining arguments are positional.
    parser.add_argument(
        "--out", 
        required=True,
        help="Path to an output directory (which will be created \
if it does not exist) to store the processed lead II data files."
    )

    parser.add_argument(
        "--ecgs", 
        required=False,
        default=Path(
            "..",
            "raw_data", 
            "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0",
            "WFDBRecords"
        ),
        help="Path to the WFDBRecords directory. Default is ../raw_data/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/WFDBRecords"
    )

    parser.add_argument(
        "--rate", 
        required=False,
        default=500,
        help="The sampling rate for the ECGs (in Hz). The \
default is 500 Hz."
    )

    parser.add_argument(
        "--pf", 
        required=False,
        default=50,
        help="Power line frequency to remove from all signals.  \
The default is to use the current (March 2024) value for China of 50 Hz."
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    sys.exit(main())