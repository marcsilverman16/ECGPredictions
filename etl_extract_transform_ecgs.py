#!/usr/bin/env python3
"""Extract and transform WFDB ECG Matlab files from PhysioNet."""

from pathlib import Path
import sys
import shutil

from scipy import io as sio
import polars as pl
import numpy as np

# https://physionet.org/content/ecg-arrhythmia/1.0.0/WFDBRecords/01/010/#files-panel
pl.enable_string_cache()
SAMPLING_FREQUENCY = 500

def main():
    input_ecg_dir = Path(
        "..",
        "raw_data", 
        "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0",
        "WFDBRecords"
    )

    output_path = Path(
        "..",
        "processed_data",
        "all_ecg_data.parquet.snappy"
    )

    # Temporary location to save individual files:
    temp_output_dir = Path(
        "..",
        "processed_data",
        "individual_ecgs"
    )

    process_ecg_files(
        input_ecg_dir=input_ecg_dir,
        sampling_freq=SAMPLING_FREQUENCY,
        temp_output_dir=temp_output_dir,
        output_path=output_path
    )
    return None

def ecg_to_polars_df(
    ecg_file_path:Path,
    sampling_freq:float,
    voltages_only=False
    ) -> pl.DataFrame:
    """Read in an ECG Matlab file.

    Args:
        ecg_file_path: path
        sampling_freq: The number of times per
            second that a voltage is recorded.
        voltages_only: If False (the default),
            then the output will contain columns
            for the patient_id and time.
            If True, then the output will
            only contain the voltages.

    Returns:
        polars.DataFrame of voltages sorted by
        time.
    """
    # Load Matlab file. 
    ecg_array = sio.loadmat(str(ecg_file_path))

    # Convert Matlab file.
    ecg_df = pl.from_numpy(
        # There is only one key
        # and that's "val".
        # It contains the ECG voltages
        # as a numpy array of dimension
        # (12) by (the sampling rate times the time).
        data=ecg_array["val"],
        # The rows in the numpy array will become
        # columns in the polars dataframe.
        orient="col",
        # I got this order from looking at a couple
        # of the header files.
        # I assume that it's the same order
        # for every patient.
        schema=[
            "lead_I",
            "lead_II",
            "lead_III",
            "lead_aVR",
            "lead_aVL",
            "lead_aVF",
            "lead_V1",
            "lead_V2",
            "lead_V3",
            "lead_V4",
            "lead_V5",
            "lead_V6"
        ]
    )

    if voltages_only:
        return ecg_df
    
    ecg_df_num_rows = ecg_df.shape[0]
    total_time = ecg_df_num_rows / sampling_freq

    # Get the patient_id from the file name.
    patient_id = ecg_file_path.stem
    ecg_df = (ecg_df
        .with_columns([
            pl.lit(value=patient_id, dtype=pl.Categorical)
                .alias("patient_id"),

            pl.Series(
                name="time",
                values=np.linspace(
                    start=0,
                    stop=total_time,
                    endpoint=False,
                    num=ecg_df_num_rows 
                )
            )
        ]) 
        # Reorder columns
        .select(
            pl.col("patient_id"),
            pl.col("time"),
            pl.col("*").exclude("patient_id", "time")
        )
    )
    return ecg_df 


def process_ecg_file(
    ecg_file_path:Path,
    sampling_freq:float,
    output_dir:Path
) -> Path:
    """Process an ECG Matlab file.
    
    Read in an ECG Matlab file and
    convert it to a polars DataFrame.
    Next, convert the polars DataFrame
    to a parquet file and write it to disk.

    Args:
        sampling_freq: The number of times per
            second that a voltage is recorded.

    Returns:
        Path to written file.
    """
    ecg_df = ecg_to_polars_df(
        ecg_file_path=ecg_file_path,
        sampling_freq=sampling_freq    
    )
    
    # Prepare to write to disk.
    # Get the patient_id from the file name.
    patient_id = ecg_file_path.stem
    output_path = output_dir / Path("".join([patient_id, ".ecg.parquet.snappy"]))

    ecg_df.write_parquet(
        file=output_path,
        compression="snappy"
    )

    return output_path

def concat_psnappy(
    input_parent_dir:Path,
    output_path:Path
) -> None:
    """Concatenate multiple .parquet.snappy files."""
    input_files = input_parent_dir.glob(
        pattern="*.parquet.snappy"
    ) 
    # Scan in a batch of .parquet.snappy files 
    # from disk
    # as polars LazyFrame's.
    # Concatenate the polars LazyFrames.
    # Write concatenated object to disk.
    (
        pl.scan_parquet(
            source=input_files,
            rechunk=True,
            low_memory=True
        )
        .sink_parquet(
            path=output_path,
            compression="snappy"
        )
    )
    return None

def process_ecg_files(
    input_ecg_dir:Path,
    sampling_freq:float,
    temp_output_dir:Path,
    output_path:Path
) -> None:
    """Process ECG Matlab files.
    
    For each file, read it in and
    convert it to a polars DataFrame.
    Next, convert the polars DataFrame
    to a parquet file and write it to disk.
    Finally, combine all of the parquet files.

    Args:
        input_ecg_dir: This Path and its
            descendants are searched
            recursively for files ending in
            ".mat"
        sampling_freq: The number of times per
            second that a voltage is recorded.
        temp_output_dir
        output_path

    Returns:
        None.
    """
    # https://docs.python.org/3/library/pathlib.html#pathlib.Path.glob
    input_ecg_paths = input_ecg_dir.glob(
        pattern="**/*.mat"
    )
    # Create temporary location to save individual files:
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    for ecg in input_ecg_paths:
        process_ecg_file(
            ecg_file_path=ecg,
            sampling_freq=sampling_freq,
            output_dir=temp_output_dir
        )

    concat_psnappy(
        input_parent_dir=temp_output_dir,
        output_path=output_path
    )

    # Clean up
    shutil.rmtree(temp_output_dir)

    return None

# https://docs.python.org/3/library/__main__.html
if __name__ == "__main__":
    sys.exit(main())