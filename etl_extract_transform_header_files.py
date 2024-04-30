#!/usr/bin/env python3
"""Extract and transform WFDB header files from PhysioNet."""

from pathlib import Path
import sys

import polars as pl

# https://physionet.org/content/ecg-arrhythmia/1.0.0/WFDBRecords/01/010/#files-panel
pl.enable_string_cache()

def main():
    wfdb_records = Path(
        "..",
        "raw_data", 
        "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0",
        "WFDBRecords"
    )

    # https://docs.python.org/3/library/pathlib.html#pathlib.Path.glob
    input_header_files = wfdb_records.glob(
        pattern="*/*/*.hea"
    )

    processed_data_path = Path("..", "processed_data")
    output_path = Path(
        processed_data_path,
        "all_header_file_data.parquet.snappy"
    )

    snomed_file = Path(processed_data_path, "corrected_snomed_mapping.csv")
    snomed_info = load_snomed_mapping(
        sm=snomed_file
    )
    
    process_header_files(
        input_header_files=input_header_files,
        snomed_info=snomed_info,
        output_path=output_path
    ) 
    
    return None

def hea_to_polars_df(
    header_file:Path
) -> pl.DataFrame:
    """Convert an ECG header file.
    
    Convert the file from WFDB format
    to a polars dataframe.

    Args:
        header_file: Path to file. 
            Assume that the file contains
            Age, Sex, and Dx.
            Note that patient_id will be taken
            from header_file.stem.

    Returns:
        polars.DataFrame with patient data
    """
    # Get the patient_id from the file name.
    patient_id = header_file.stem

    # Read lines
    with open(file=header_file, mode="r") as hf_handler:
        for line in hf_handler:
            if line.startswith("#Age"):
                # We have identified the part of
                # the file with the good data.
                age = (line
                    .rstrip()
                    .rsplit(
                        maxsplit=1
                    )[1]
                )
                try:
                    age = int(age)
                except ValueError:
                    age = None
            
                line = hf_handler.readline()
                sex = (line
                    .rstrip()
                    .rsplit(
                        maxsplit=1
                    )[1]
                )

                line = hf_handler.readline()
                diagnoses = (line
                    .rstrip()
                    .rsplit(
                        maxsplit=1
                    )[1].rsplit(sep=",")
                )
    
    # Create polars dataframe with patient info.
    header_df = pl.DataFrame(
        data=[
            pl.Series(name="patient_id", values=[patient_id], dtype=pl.Categorical),
            pl.Series(name="age", values=[age], dtype=pl.UInt8),
            pl.Series(name="sex", values=[sex], dtype=pl.Categorical),
            # Assume that all of the SNOMED
            # Concept IDs are integers.
            pl.Series(name="diagnoses", values=[diagnoses], dtype=pl.List(pl.UInt64))
        ],
        schema={
            "patient_id": pl.Categorical,
            "age": pl.UInt8,
            "sex": pl.Categorical,
            "diagnoses": pl.List(pl.UInt64)
        }
    )
    # Explode list of diagnoses
    # into new rows for join
    header_df = (header_df
        .explode("diagnoses")
    )
    return header_df
    
def load_snomed_mapping(sm:Path) -> pl.DataFrame:
    """Load .csv file into polars DataFrame"""
    return(
        pl.read_csv(
            source=sm,
            dtypes={
                "snomed_concept_code": pl.UInt64, 
                "snomed_concept_name": pl.Categorical
            }
        )
    )

def join_hea_with_snomed(
    hea_df:pl.DataFrame, 
    snomed_info:pl.DataFrame, 
    unrecognized_snomed_concept_codes:pl.Series
) -> dict:
    """Appends to unrecognized_snomed_concept_codes
    
    as needed.

    Returns:
        dict: with keys of unrecognized_snomed_concept_codes, 
            hea_df_with_snomed
    """
    hea_df_with_snomed = (hea_df
        .join(
            other=snomed_info,
            left_on="diagnoses",
            right_on="snomed_concept_code",
            how="inner"
        )
        # The codes are not needed because
        # we have the names now.
        .drop("diagnoses")
        # Reorder columns.
        .select(
            pl.col("*").exclude("snomed_concept_name"),
            "snomed_concept_name"
        )
    )

    # Although we do an inner join above,
    # there may be diagnoses without matching
    # snomed_concept_code's.
    # So, check for that now.
    if hea_df_with_snomed.shape[0] == 0:
        # There was no matching code.
        unrecognized_snomed_concept_code_for_patient = (hea_df
            .select("diagnoses")
            .to_series()
        )
        # Add it to the list of unrecognized codes.
        (unrecognized_snomed_concept_codes
            .append(unrecognized_snomed_concept_code_for_patient)
        )
    
        
    return(
        {
            "unrecognized_snomed_concept_codes": unrecognized_snomed_concept_codes,
            "hea_df_with_snomed": hea_df_with_snomed
        }
    )

def process_header_file(
    hf:Path, 
    snomed_info:pl.DataFrame, 
    unrecognized_snomed_concept_codes:pl.Series
) -> dict:
    """Extract and transform header file.
    
    Extract info. from the header file to make 
    a polars DataFrame.  Join with snomed_info.
    
    Args:
        hf: Path to header file
        snomed_info: polars DataFrame for join

    Returns:
        dict with keys of unrecognized_snomed_concept_codes
            and hea_df_with_snomed
    """
    hea_df = hea_to_polars_df(header_file=hf)
    
    hea_with_snomed_dict = join_hea_with_snomed(
        hea_df=hea_df,
        snomed_info=snomed_info,
        unrecognized_snomed_concept_codes=unrecognized_snomed_concept_codes
    )

    return hea_with_snomed_dict

def process_header_files(
    input_header_files, 
    snomed_info:pl.DataFrame, 
    output_path:Path
) -> None:
    """Extract and transform header files.
    
    Extract info. from each header file to make 
    a polars DataFrame.  Join with snomed_info.
    Concatenate the separate DataFrames.
    Write Parquet file to disk.
    
    Args:
        input_header_files: generator of Path's
        snomed_info: polars DataFrame for join
        output_path
        
    Returns:
        None
    """
    unrecognized_snomed_concept_codes = pl.Series(name="diagnoses")
    
    combined_hea_file_data_df = pl.DataFrame()
    for hf in input_header_files:
        processing_results = process_header_file(
            hf=hf,
            snomed_info=snomed_info,
            unrecognized_snomed_concept_codes=unrecognized_snomed_concept_codes
        )
        # Unpack processing_results
        hea_df_with_snomed = processing_results["hea_df_with_snomed"]
        unrecognized_snomed_concept_codes = processing_results["unrecognized_snomed_concept_codes"]

        # We can stop processing all of the files
        # as soon as we have some unrecognized_snomed_concept_codes.
        if unrecognized_snomed_concept_codes.len() > 0:
            print("Unrecognized snomed_concept_codes:")
            # Inform user of problems.
            print(unrecognized_snomed_concept_codes)
            return None

        combined_hea_file_data_df = pl.concat([combined_hea_file_data_df, hea_df_with_snomed])
  
    

    # Transform if there were no unrecognized_snomed_concept_codes
    combined_hea_file_data_df.write_parquet(
        file=output_path,
        compression="snappy"
    )

    return None

# https://docs.python.org/3/library/__main__.html
if __name__ == "__main__":
    sys.exit(main())