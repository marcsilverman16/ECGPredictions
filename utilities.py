#!/usr/bin/env python3
"""Utility code for the project."""

from pathlib import Path
import itertools

from scipy import io as sio
import polars as pl
import numpy as np

def read_sample_of_header_info(hf:Path|str, proportion_wanted:float) -> pl.DataFrame:
    """Read a sample of rows from a .parquet.snappy file.
    
    The file must contain a column called 'patient_id',
    which will be used to sort the result.
    """
    # Read in data
    entire_thing_df = pl.read_parquet(
        source=hf
    )

    # Now, figure out which data we are taking.
    our_sample = (entire_thing_df
        .sample(
            fraction=proportion_wanted, 
            with_replacement=False,
            shuffle=True
        )
        # Sampling can mess with the order,
        # so redo the sorting.
        # https://stackoverflow.com/a/76995707/8423001
        .sort(by=pl.col("patient_id"))
    )

    return our_sample


def partition_by_sex_age(
    hf:pl.LazyFrame|pl.DataFrame,
    age_groups:list[tuple],       
    ) -> list[tuple[str, str, pl.LazyFrame|pl.DataFrame]]:
    """Partition a polars frame by sex and age.
    
    Args:
        hf: polars.LazyFrame or polars.DataFrame. 
            Contains data from the header files 
            of multiple patients. Contains columns of "sex"
            and "age".
        age_groups: Tuples defining the closed intervals for age.
    """
    # First, figure out what the sex groups are.
    if isinstance(hf, pl.LazyFrame):
        sex_groups = (hf
            .select("sex")
            .unique()
            .collect()
            .to_series()
        )
    else:
        # hf is a DataFrame, so
        # don't collect.
        sex_groups = (hf
            .select("sex")
            .unique()
            .to_series()
        )

    # Prepare for loop.
    # filtered_lfs will be filled in loop.
    # Each element in filtered_lfs will be a 
    # polars LazyFrame with only one sex and age group.
    filtered_frames = []
    for sex in sex_groups:
        for age_group in age_groups:
            filtered_frame = (hf
                .filter(
                    (pl.col("sex") == sex)
                    &
                    # Use * to unpack the tuple of age_groups
                    # to get the lower and upper bounds.
                    (pl.col("age").is_between(*age_group))
                )
                .select(
                    pl.exclude("sex", "age")
                )
            )

            # Construct a tuple to add to the list
            # that this function will return.
            current_group = (
                "_".join(["sex", sex.lower()]),
                "_".join(["age", str(age_group[0]), "to", str(age_group[1])]),
                filtered_frame
            )
            filtered_frames.append(current_group)
   
        # We have finished going through all of the 
        # numeric age groups.
        # Finish off by handling nulls for age
        # within each sex.
        filtered_frame = (hf
            .filter(
                (pl.col("sex") == sex)
                &
                (pl.col("age").is_null())
            )
            .select(
                pl.exclude("sex", "age")
            )
        )

        # Construct a tuple to add to the list
        # that this function will return.
        current_group = (
            "_".join(["sex", sex.lower()]),
            "age_null",
            filtered_frame
        )
        filtered_frames.append(current_group)
            
    return filtered_frames


def get_intervals(
        start:int|float, 
        lengths:int|float|list|np.ndarray, 
        spacings:int|float|list|np.ndarray=None,
        num_intervals:int=None
    ):
    """Get a list of tuples where each tuple represents an interval.

    The intervals are degenerate whenever a length in lengths is 0.
    The intervals can be constructed for the real number 
    line or the integers, where the latter is the default.

    Example: get_intervals(start=0, lengths=[1, 2, 0, 0, 3], spacings=1)
        gives
        [(0, 1), (2, 4), (5, 5), (6, 6), (7, 10)]

    Args:
        start: A greatest lower bound for all of our intervals.
        lengths: The lengths of each interval. If this is a 
            single number and not a list, then the lengths
            of each interval are taken to all be the same.
        spacings: The distances between intervals. If this is a 
            single number and not a list, then the spacings
            between each interval are taken to all be the same.
            Default is 1.
        num_intervals: The number of intervals. Default is
            len(lengths)
    """
    # Handle arguments.
    # Ravel to ensure that we have 1-d arrays.
    if spacings is None:
        spacings = 1

    start = np.array(start).ravel()
    lengths = np.array(lengths).ravel()
    spacings = np.array(spacings).ravel()

    # if np.any(spacings < 0):
    #     raise RuntimeError("You cannot have negative spacings.")
    if np.any(lengths < 0):
        raise RuntimeError("You cannot have negative lengths.")

    num_lengths = len(lengths)
    num_spacings = len(spacings)

    if num_lengths == 1:
        if num_spacings == 1 and num_intervals is not None:
            lengths = np.broadcast_to(array=lengths, shape=(num_intervals,))
            spacings = np.broadcast_to(array=spacings, shape=(num_intervals - 1,))
        else:
            raise RuntimeError
    else:
        if num_spacings == 1:
            if num_intervals is not None and num_intervals != num_lengths:
                raise RuntimeError("num_intervals should be the same as the number of lengths.")
            else:
                spacings = np.broadcast_to(array=spacings, shape=(num_lengths - 1,))
        else:
            if num_intervals is not None:
                if num_intervals != num_lengths or (num_spacings + 1) != num_lengths:
                    raise RuntimeError
            else:
                if (num_spacings + 1) != num_lengths:
                    raise RuntimeError

    # Implement main logic of function
    spacings = np.concatenate(([0], np.array(spacings)))
    total_dis = np.array(lengths) + spacings
    b = start + total_dis.cumsum()
    a = b - lengths

    return [(_a, _b) for _a, _b in zip(a, b)]
    

def make_nested_dirs(
    *args 
    ) -> list[Path]:
    """Make nested directories.

    Args:
        *args: Additional arguments which are
            lists of strings. 
            Each string is the name of a child directory
            in the hierarchy that will be created.
            For example, an input of ["a", "b"] as
            one positional argument followed by ["c", "d"]
            as another positional argument will cause 
            the following to be created:

            ./
            ├── a/
            │   ├── c
            │   └── d
            └── b/
                ├── c
                └── d
            created by: tree.nathanfriend.io
    """
    paths_to_make = [Path(*p) for p in itertools.product(*args)]
    for p in paths_to_make:
        # https://stackoverflow.com/a/50110841/8423001
        # Make the directory if it doesn't already exist.
        # If it already exists, do nothing.
        p.mkdir(parents=True, exist_ok=True)

    return paths_to_make


def read_matrix(matlab_file_path:Path|str) -> np.ndarray:
    """Read a matrix from a .mat file.
    
    Assume that the MATLAB file contains a single matrix
    for a variable called 'val'.  Also, assume that the
    matrix needs to be transposed.

    Args:
        matlab_file_path
    Returns:
        numpy.ndarray
    """
    return sio.loadmat(str(matlab_file_path))["val"].T


def get_wanted_file_paths(parent_dir:Path|str, file_stems:list[str], suffix=".mat") -> list[Path]:
    """Get sorted list of file paths in parent_dir   
    
    and its subdirectories that match one of the
    provided file_stems and have the same suffix.
    See: https://docs.python.org/3/library/pathlib.html#pathlib.PurePath.name

    Args:
        parent_dir: This directory and its subdirectories
            are searched.
        file_stems: A sorted list (in ascending order) of
            unique strings for the file names without
            the suffix.
        suffix: End of each file name.

    Returns:
        Ordered list of file paths with matching file_stems.
        The file paths will be relative to parent_dir.
        The returned file paths are sorted according to
        their stems in ascending order so that they match
        up with the argument file_stems.
    """
    search_pattern = "".join(["**/", "*", suffix])
    # First, find everything with the given suffix
    # because that's the common element.
    the_parent_dir = Path(parent_dir)
    found_file_paths_generator = the_parent_dir.glob(pattern=search_pattern)
    found_file_paths_list = list(found_file_paths_generator)
    # Globbing finds file paths that are not sorted.
    # We need to look at the stems of the paths to
    # figure out how to sort them.
    # found_stems_indices_to_sort = np.argsort(found_stems)

    # Convert file_stems into a frozenset so that
    # it is easy to determine if a given element is within it.
    wanted_stems = frozenset(file_stems)

    # prep for loop
    found_and_wanted_stems = []
    found_and_wanted_file_paths = []
    for file_path in found_file_paths_list:
        found_stem = file_path.stem

        if found_stem in wanted_stems:
            found_and_wanted_stems.append(found_stem)
            found_and_wanted_file_paths.append(file_path)
        
    found_and_wanted_file_paths_sorted_tuple = sorted(
        zip(found_and_wanted_file_paths, found_and_wanted_stems),
        key=lambda path_index_tuple: path_index_tuple[1]
    )

    found_and_wanted_file_paths_sorted = [pth for pth, path_index in found_and_wanted_file_paths_sorted_tuple]

    return found_and_wanted_file_paths_sorted


def get_leads(*names_of_leads, ecg:np.ndarray) -> np.ndarray:
    """Get leads from the ECG by name.
    
    Assume that leads correspond to columns.

    Args:
        names_of_leads: Positional arguments of names
            of leads as strings.
    """
    # I got this order from looking at a couple
    # of the header files.
    # I assume that it's the same order
    # for every patient.
    ecg_name_mapping = {
        "I": 0,
        "II": 1,
        "III": 2,
        "aVR": 3,
        "aVL": 4,
        "aVF": 5,
        "V1": 6,
        "V2": 7,
        "V3": 8,
        "V4": 9,
        "V5": 10,
        "V6": 11
    }
    # https://stackoverflow.com/a/55395697/8423001
    col_indices = [ecg_name_mapping[lead] for lead in names_of_leads]
    
    out = np.take(
        a=ecg,
        indices=col_indices,
        axis=1
    )

    return out
    