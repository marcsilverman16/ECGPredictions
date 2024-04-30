#!/usr/bin/env python3
"""Calculate dissimilarity matrix for ECG data using DTW.

The main function is write_dissimilarity_matrices.

Given a list of patient_ids, 
get the ECGs for all of the possible pairs
and feed each pair one at a time
into the DTW function.  Save the DTW
dissimilarity into the right
spot in the dissmilarity matrix.
"""

import sys
from utilities import \
    read_matrix, \
    make_nested_dirs, \
    get_intervals, \
    partition_by_sex_age
    
from pathlib import Path
import argparse
from multiprocessing import Pool
import timeit

from scipy import io as sio
import polars as pl
import numpy as np
from tslearn.metrics import dtw

def main():
    user_args = get_user_args()
    path_to_input_info = Path(user_args.inp)
    path_to_r_peak_info = Path(user_args.r)
    dissimilarity_output = Path(user_args.dis)
    try:
        min_num_patients = int(user_args.min)
    except:
        min_num_patients = None
    try:
        max_num_patients = int(user_args.max)
    except:
        max_num_patients = None

    strata_proportion = float(user_args.stp)
    global_constraint = user_args.global_constraint

    try:
        sakoe_chiba_radius = round(float(user_args.sakoe_chiba_radius))
    except:
        sakoe_chiba_radius = None
    
    try:
        itakura_max_slope = float(user_args.itakura_max_slope)
    except:
        itakura_max_slope = None

    try:
        age_groups = eval(user_args.age_groups)
    except:
        age_groups = None
        print("Using default age_groups.")

    ###################################################
    r_peak_info = pl.scan_parquet(
        path_to_r_peak_info
    )

    ordered_r_peaks = (r_peak_info
        .select("r_peak")
        .collect()
        .to_series()                   
    )    

    write_dissimilarity_matrices(
        path_to_input_info=path_to_input_info,
        dissimilarity_output=dissimilarity_output,
        ordered_r_peaks=ordered_r_peaks,
        min_num_patients=min_num_patients,
        max_num_patients=max_num_patients,
        strata_proportion=strata_proportion,
        age_groups=age_groups,
        global_constraint=global_constraint,
        sakoe_chiba_radius=sakoe_chiba_radius,
        itakura_max_slope=itakura_max_slope
    )
        
    return None


def write_dissimilarity_matrices(
    path_to_input_info:str|Path,
    dissimilarity_output:str|Path,
    ordered_r_peaks:pl.Series,
    min_num_patients:int=None,
    max_num_patients:int=None,
    strata_proportion:float=1.0,
    age_groups:list[tuple]=None,
    global_constraint:str=None,
    sakoe_chiba_radius:int=None,
    itakura_max_slope:float=None
    ) -> None:
    """Write dissimilarity matrices for age/sex strata."""
    
    ################################################################
    # Handle args.
    ################################################################
    path_to_input_info = Path(path_to_input_info)
    dissimilarity_output = Path(dissimilarity_output)

    if age_groups is None:
        # Get the default for age_groups.
        lengths = [0 for i in range(15)]
        lengths.extend([4 for i in range(15)])

        age_groups = get_intervals(
            start=0, 
            lengths=0,
            spacings=1,
            num_intervals=15
        )

        age_groups.extend(
            get_intervals(
                start=15, 
                lengths=4,
                spacings=1,
                num_intervals=15
            )
        )
    if min_num_patients is None:
        min_num_patients = 3

    ###########################################################
    input_info_lf = pl.scan_parquet(
        source=path_to_input_info
    )

    input_info_grouped_list = partition_by_sex_age(
        hf=input_info_lf,
        age_groups=age_groups
    )

    # Loop through input_info_grouped_list.
    # In each iteration, check if a non-empty frame
    # exists.  If it does, then make the corresponding
    # nested dir for the sex and age_group.
    # Parse out the patient_ids and ecg_paths and
    # feed them into fill_and_write_dissimilarity_matrix.
    for sex_str, age_group_str, input_info_grouped in input_info_grouped_list:
        input_info_grouped = input_info_grouped.collect()
        
        # https://stackoverflow.com/questions/75523498/python-polars-how-to-get-the-row-count-of-a-dataframe
        # How many patients do we have within this sex and age_group?
        num_patients = input_info_grouped.select(pl.len()).item()
        if max_num_patients is None:
            max_num_patients = num_patients
        # How should we sample?
        possible_sample_size = round(strata_proportion * num_patients)
        # How does possible_sample_size compare to our caps?
        if possible_sample_size < min_num_patients:
            possible_sample_size = min(min_num_patients, num_patients)
        elif possible_sample_size > max_num_patients:
            possible_sample_size = min(max_num_patients, num_patients)
        
        if possible_sample_size < 2:
            # We are seeking less than 2 patients.
            # It doesn't make sense to do DTW.
            continue

        # Take a sample.
        input_info_grouped_sampled = (input_info_grouped
            .sample(
                n=possible_sample_size, 
                with_replacement=False,
                shuffle=True
            )
        )

        # Sampling can mess with the order,
        # so redo the sorting.
        # https://stackoverflow.com/a/76995707/8423001
        input_info_grouped_sampled = (
            input_info_grouped_sampled
            .sort(by=pl.col("patient_id"))
        )

        nested_dir = make_nested_dirs(
            [dissimilarity_output],
            [sex_str],
            [age_group_str]
        )[0]

        output_path = Path(nested_dir, "dissimilarity_matrix.parquet.snappy")

        patient_ids = (input_info_grouped_sampled
            .select(pl.col("patient_id"))
            .to_series()
            .to_list()
        )

        ecg_paths = (input_info_grouped_sampled
            .select(pl.col("ecg_path"))
            .to_series()
            .to_list()
        )

        print(f"Calculating dissimilarity matrix for {possible_sample_size} patients in {sex_str} and {age_group_str} . . .")
        
        fill_and_write_dissimilarity_matrix(
            ordered_patient_ids=patient_ids,
            ordered_paths_to_observations=ecg_paths,
            dissimilarity_output=output_path,
            ordered_r_peaks=ordered_r_peaks,
            global_constraint=global_constraint,
            sakoe_chiba_radius=sakoe_chiba_radius,
            itakura_max_slope=itakura_max_slope
        )

    return None


def fill_and_write_dissimilarity_matrix(
    ordered_patient_ids:list[str],
    ordered_paths_to_observations:list[Path|str],  
    dissimilarity_output:Path|str,
    ordered_r_peaks:pl.Series,
    # https://realpython.com/python-kwargs-and-args/
    **kwargs
    ) -> None:
    # # First, figure out which data we are taking.
    # wanted_patient_ids = read_patient_ids(
    #     path_to_patient_ids=path_to_patient_ids,
    #     proportion_wanted=proportion_wanted
    # ).to_list()
    
    start_time = timeit.default_timer()
    # wanted_file_paths = get_wanted_file_paths(
    #     parent_dir=input_ecg_dir,
    #     file_stems=wanted_patient_ids,
    #     suffix=".mat"
    # )
    
    
    dissimilarity_matrix_lf = fill_dissimilarity_matrix(
        ordered_paths_to_observations=ordered_paths_to_observations,
        ordered_patient_ids=ordered_patient_ids,
        ordered_r_peaks=ordered_r_peaks,
        **kwargs
    )

    dissimilarity_matrix_lf.sink_parquet(
        path=dissimilarity_output,
        compression="snappy"
    )
    delta_time = timeit.default_timer() - start_time
    print(f"Time elapsed within fill_and_write_dissimilarity_matrix func.: {delta_time}")


    return None

def fill_dissimilarity_matrix(
        ordered_paths_to_observations:list[Path|str],  
        ordered_patient_ids:list[str],
        ordered_r_peaks,
        # https://realpython.com/python-kwargs-and-args/
        **kwargs
    ) -> pl.LazyFrame:
    """Return the dissimilarity matrix.
    
    Args:
        ordered_paths_to_observations: list. 
            ordered_paths_to_observations should be ordered
            the same as ordered_patient_ids (and it should
            be the same length).
            Each entry is a file path
            that contains data that can be read
            by the function read_matrix.
            The length of this list will
            be the dimension of the dissimilarity matrix.
        ordered_patient_ids:
        ordered_r_peaks: sorted polars Series of lists. 
            The Series should be sorted so that the ith 
            element corresponds to the ith patient in 
            ordered_patient_ids.

        kwargs: named arguments to the get_dissimilarity
            function.

    Returns:
        numpy.ndarray of size n by n.  The (i, j)
        entry in the matrix is the dissimilarity between
        observations i and j. 
    """
    # print("ordered_r_peaks")
    # print(ordered_r_peaks)
    # print('kwargs["global_constraint"]')
    # print(kwargs["global_constraint"])
    n = len(ordered_paths_to_observations)
    dissimilarity_matrix = np.empty(shape=(n, n), dtype=np.float64)
    
    # Loop through the upper triangle of the dissimilarity_matrix
    # and fill it in.
    # https://www.educative.io/answers/what-is-the-python-timeit-module
    # start_time = timeit.default_timer()
    for i in range(n):
        ith_observation = read_matrix(ordered_paths_to_observations[i])
        ith_r_peaks = ordered_r_peaks[i]
        for j in range(i + 1, n, 1):
            jth_observation = read_matrix(ordered_paths_to_observations[j])
            jth_r_peaks = ordered_r_peaks[j]
            # https://stackoverflow.com/questions/9867562/pass-kwargs-argument-to-another-function-with-kwargs
            dissimilarity_matrix[i, j] = get_dissimilarity(
                a=ith_observation,
                a_cuts=ith_r_peaks,
                b=jth_observation,
                b_cuts=jth_r_peaks,
                **kwargs
            )
    # delta_time = timeit.default_timer() - start_time
    # print(f"Time elapsed for loop in fill_dissimilarity_matrix: {delta_time}")

    # Make the dissimilarity_matrix symmetric.
    dissimilarity_matrix= 0.5 * (dissimilarity_matrix + dissimilarity_matrix.T)
    # Make the dissimilarity_matrix have 0s along the main diagonal.
    np.fill_diagonal(dissimilarity_matrix, 0)

    # Make sure the columns are labeled for easier reference later.
    dissimilarity_matrix_lf = pl.LazyFrame(
        data=dissimilarity_matrix,
        schema=ordered_patient_ids
    )

    return dissimilarity_matrix_lf
            

def get_dissimilarity(a, a_cuts, b, b_cuts, **kwargs):
    """Calls dtw.

    Args:
        a_cuts: Cutoff indices to partition a
        b_cuts: Cutoff indices to partition b
    """
    # Partition a and b.
    # Calculate the DTW dissimilarities between all possible
    # pairs of partitions of a and b (excluding the ending
    # partitions).
    # Return the mean.
    
    num_a_segments = len(a_cuts) - 1
    num_b_segments = len(b_cuts) - 1
    num_comparisons = num_a_segments * num_b_segments
    # mini_dissimilarities will be filled later.
    mini_dissimilarities = np.empty(shape=(num_comparisons,), dtype=np.float64)

    mdi = 0
    for i in range(num_a_segments):
        # We get a slice with the value and the next value
        # of a_cuts for iteration i.
        # We add 1 because we want to include the endpoint.
        a_segment = a[slice(a_cuts[i], a_cuts[i + 1] + 1), :]
        for j in range(num_b_segments):
            b_segment = b[slice(b_cuts[j], b_cuts[j + 1] + 1), :]
            mini_dissimilarities[mdi] = dtw(
                s1=a_segment,
                s2=b_segment,
                **kwargs
            )
            mdi += 1

    d = np.mean(mini_dissimilarities)
    return d

def get_user_args() -> argparse.Namespace:
    """Get arguments from the command line."""
    parser = argparse.ArgumentParser()
    # Optional arguments are prefixed by single or double dashes.
    # The remaining arguments are positional.
    parser.add_argument("--inp", required=True, \
        help="Path to a .parquet file of containing column headers \
of 'patient_id', 'age', 'sex', 'snomed_concept_name' and 'ecg_path'. \
Each row should be for a unique patient.")
    
    parser.add_argument("--r", required=True, \
        help="Path to a .parquet file containing column headers \
of 'patient_id' and 'r_peaks'. \
Each row should be for a unique patient.")

    parser.add_argument(
        "--dis", 
        required=True,
        help="A path to a directory (which will be created) to store the dissimilarity matrices. \
The dissimilarity matrices will be stored as snappy compressed parquet files."
    )

    parser.add_argument(
        "--stp", 
        required=False,
        default=1,
        help="The common proportion of patients to sample within each \
strata of sex and age to do the dynamic time warping for.  \
For speed, \
it can be set to a number less than 1."
    )

    parser.add_argument(
        "--min", 
        required=False,
        default=3,
        help="The minimum number of patients within a \
strata of sex and age to do the dynamic time warping for. \
For example, if we request a proportion of 0.01 patients, \
but some sex and age group has a size less than 'min', then \
'min' patients will be selected from that sex and age group."
    )

    parser.add_argument(
        "--max", 
        required=False,
        default=None,
        help="The maximum number of patients within a \
strata of sex and age to do the dynamic time warping for. \
For example, if we request a proportion of 0.01 patients, \
but some sex and age group has a size greater than 'max', then \
'max' patients will be selected from that sex and age group."
    )

    parser.add_argument(
        "--age_groups", 
        required=False,
        default=None,
        help="List of tuples where each tuple represents an age group. \
For example: enter '[(0, 49), (50, 99)]' with the brackets to indicate \
age groups of 0 to 49 (all inclusive) and 50 to 99 (all inclusive). \
The default is to have single ages from 0 to 14 and then groups of \
5 years until 89."
    )

    parser.add_argument(
        "--global_constraint", 
        required=False,
        default=None,
        help="One of {'itakura', 'sakoe_chiba'} (without quotes). \
(default: None) \
Global constraint to restrict admissible paths for DTW."
    )

    parser.add_argument(
        "--sakoe_chiba_radius", 
        required=False,
        default=None,
        help="int or None (default: None) \
Radius to be used for Sakoe-Chiba band global constraint. \
If None and global_constraint is set to 'sakoe_chiba', \
a radius of 1 is used. If both sakoe_chiba_radius and \
itakura_max_slope are set, global_constraint is used to \
infer which constraint to use among the two. In this case, \
if global_constraint corresponds to no global constraint, \
a RuntimeWarning is raised and no global constraint is used. \
See tslearn documentation for more information."
    )

    parser.add_argument(
        "--itakura_max_slope", 
        required=False,
        default=None,
        help="float or None (default: None) \
Maximum slope for the Itakura parallelogram constraint. If None and \
global_constraint is set to 'itakura', a maximum slope of 2. is \
used. If both sakoe_chiba_radius and itakura_max_slope are set, \
global_constraint is used to infer which constraint to use among \
the two. In this case, if global_constraint corresponds to no global \
constraint, a RuntimeWarning is raised and no global constraint is \
used."
    )


    args = parser.parse_args()

    return args

if __name__ == "__main__":
    sys.exit(main())