#!/usr/bin/env python3
"""Generate .parquet.snappy files showing how patient_ids will be used.

The main function is write_splits.
"""
from utilities import get_wanted_file_paths

from pathlib import Path
import argparse
import sys

import polars as pl
from sklearn.model_selection import KFold

pl.enable_string_cache()

def main():
    user_args = get_user_args()
    num_outer_folds = int(user_args.num_outer_folds)
    num_inner_folds = int(user_args.num_inner_folds)
    all_header_file_data_path = Path(user_args.all_header_file_data_path)
    split_info_out = Path(user_args.split_info_out)
    ecg_dir = Path(user_args.ecgs)

    write_splits(
        num_outer_folds=num_outer_folds,
        num_inner_folds=num_inner_folds,
        all_header_file_data_path=all_header_file_data_path,
        ecg_dir=ecg_dir,
        split_info_out=split_info_out
    )

    return None


def write_splits(
        num_outer_folds:int, 
        num_inner_folds:int,
        all_header_file_data_path:Path|str=Path("..", "processed_data", "all_header_file_data.parquet.snappy"),
        ecg_dir:Path|str=Path("../raw_data/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/WFDBRecords"),
        split_info_out:Path|str=Path("..", "processed_data", "split_info")
    ):
    """Read in the data at all_header_file_data_path 
    
    and split it up for nested cross validation.  
    Write each split to disk.
    
    Args:
    
    Returns:
        None

    Side Effects:
        Creates directories and files within the  
        directory of split_info_out.
        Also, prints to stdout.
    """
    # Read from all_header_file_data_path
    all_header_file_data = pl.scan_parquet(all_header_file_data_path)

    # Create a list column out of snomed_concept_name.
    all_header_file_data = (all_header_file_data
        .group_by(pl.all().exclude("snomed_concept_name"))
        .agg(pl.col("snomed_concept_name"))
        .select(
            pl.col("patient_id").cast(pl.Utf8),
            pl.all().exclude("patient_id")
        )
        .sort(by="patient_id")
    )

    # Determine file paths to ECGs.
    sorted_patient_ids = (all_header_file_data
        .select(pl.col("patient_id"))
        .sort(by="patient_id")
        .collect()
        .to_series()
        .to_list()
    )
    
    ecg_paths = get_wanted_file_paths(
        parent_dir=ecg_dir,
        file_stems=sorted_patient_ids
    )

    ecg_path_strings = [str(p) for p in ecg_paths]

    X_shuffled = (all_header_file_data
        .with_columns([
            pl.Series(name="ecg_path", values=ecg_path_strings, dtype=pl.Utf8)
        ]) 
        .with_row_index()
        .collect()
        # Calling sample here allows us
        # to shuffle the polars DataFrame.
        .sample(
            fraction=1, 
            with_replacement=False, 
            shuffle=True
        )                       
    )

    # Make split_info_out into a Path for sure.
    split_info_out_path = Path(split_info_out)
    extension = ".parquet.snappy"

    num_patients = X_shuffled.height
    print(f"We have data on {num_patients} patients.")

    outer_kfold_obj = KFold(n_splits=num_outer_folds, shuffle=True)
    outer_fold_splittings = outer_kfold_obj.split(X=X_shuffled)
    for outer_fold_index, (indices_for_later, test_indices) in enumerate(outer_fold_splittings):
        # https://stackoverflow.com/a/50110841/8423001
        outer_fold_dir = Path(
            split_info_out_path,
            "".join(["fold_", str(outer_fold_index)])
        )
        # Make the directory if it doesn't already exist.
        # If it already exists, do nothing.
        outer_fold_dir.mkdir(parents=True, exist_ok=True)
        outer_fold_testing_dir = Path(outer_fold_dir, "testing")
        outer_fold_testing_dir.mkdir(parents=True, exist_ok=True)
        outer_fold_testing_file_path = Path(
            outer_fold_testing_dir,
            "".join(["testing", extension])
        )

        testing_info = (X_shuffled
            .filter(pl.col("index").is_in(test_indices))
            .select(pl.exclude("index"))
            .sort(
                by=pl.col("patient_id")
            )
        )
 
        print(f"For outer fold {outer_fold_index}, we have {testing_info.height} patients in the testing set.")
        testing_info.write_parquet(file=outer_fold_testing_file_path, compression="snappy")
        # Get things set up for nested k-fold cross validation.
        # Now, get things set up for k-fold cross validation
        # within each fold of the outer fold, so that we can choose
        # a good model with good hyperparameters for the outer fold.
        inner_kfold_obj = KFold(n_splits=num_inner_folds, shuffle=True)
        outer_fold_combined_training_and_validation = (X_shuffled
            .filter(pl.col("index").is_in(indices_for_later))
            # Remake the index.
            .drop("index")
            .with_row_index()
        )

        inner_fold_splittings = inner_kfold_obj.split(
            X=outer_fold_combined_training_and_validation
        )
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold.get_n_splits
        for inner_fold_index, (training_indices, validating_indices) in enumerate(inner_fold_splittings):
            # Caution: training_indices and validating_indices
            # index outer_fold_combined_training_and_validation
            # and not X_shuffled!
            # https://stackoverflow.com/a/50110841/8423001
            inner_fold_dir = Path(
                outer_fold_dir, 
                "".join(["fold_", str(inner_fold_index)])
            )
            # Make the directory if it doesn't already exist.
            # If it already exists, do nothing.
            inner_fold_dir.mkdir(parents=True, exist_ok=True)

            inner_fold_training_dir = Path(inner_fold_dir, "training")
            inner_fold_training_dir.mkdir(parents=True, exist_ok=True)
            inner_fold_training_file_path = Path(
                inner_fold_training_dir, 
                "".join(["training", extension])
            )
            
            training_info = (outer_fold_combined_training_and_validation
                .filter(pl.col("index").is_in(training_indices))
                .select(pl.exclude("index"))
                .sort(
                    by=pl.col("patient_id")
                )
            )
            print(f"Within outer fold {outer_fold_index} and inner fold {inner_fold_index}, we have {training_info.height} patients in the training set.")
            training_info.write_parquet(file=inner_fold_training_file_path, compression="snappy")

            inner_fold_validating_dir = Path(inner_fold_dir, "validating")
            inner_fold_validating_dir.mkdir(parents=True, exist_ok=True)
            inner_fold_validating_file_path = Path(
                inner_fold_validating_dir,
                "".join(["validating", extension])
            )

            validating_info = (outer_fold_combined_training_and_validation
                .filter(pl.col("index").is_in(validating_indices))
                .select(pl.exclude("index"))
                .sort(
                    by=pl.col("patient_id")
                )
            )
            print(f"Within outer fold {outer_fold_index} and inner fold {inner_fold_index}, we have {validating_info.height} patients in the validating set.")
            validating_info.write_parquet(file=inner_fold_validating_file_path, compression="snappy")

def get_user_args() -> argparse.Namespace:
    """Get arguments from the command line."""
    parser = argparse.ArgumentParser()
    # Optional arguments are prefixed by single or double dashes.
    # The remaining arguments are positional.
    parser.add_argument("--num_outer_folds", required=True, \
        help="Positive integer.  The number of outer folds in the \
nested cross validation.")
    
    parser.add_argument(
        "--num_inner_folds", 
        required=True,
        help="Positive integer.  The number of inner folds in the \
nested cross validation."
    )

    parser.add_argument(
        "--split_info_out", 
        required=False,
        default=Path("..", "processed_data", "split_info"),
        help="A path to a directory (which will be created) \
to store the information on the splits."
    )

    parser.add_argument(
        "--all_header_file_data_path", 
        required=False,
        default=Path("..", "processed_data", "all_header_file_data.parquet.snappy"),
        help="The path to the .parquet file containing all of the \
the header information for all of the patients."
    )

    parser.add_argument(
        "--ecgs", 
        required=False,
        default=Path("../raw_data/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/WFDBRecords"),
        help="The path to the directory containing the ECG files."
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    sys.exit(main())