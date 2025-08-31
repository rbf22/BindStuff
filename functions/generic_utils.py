"""Generic utility functions for the BindStuff pipeline."""
####################################
################## General functions
####################################
### Import dependencies
import os
import json
import shutil
import sys
import zipfile
from typing import Dict, List, Tuple, Any
import jax
import pandas as pd
import numpy as np

# Define labels for dataframes
def generate_dataframe_labels() -> Tuple[List[str], List[str], List[str]]:
    """Generate column labels for trajectory dataframe."""
    # labels for trajectory
    trajectory_labels = [
        "Design",
        "Protocol",
        "Length",
        "Seed",
        "Helicity",
        "Target_Hotspot",
        "Sequence",
        "InterfaceResidues",
        "pLDDT",
        "pTM",
        "i_pTM",
        "pAE",
        "i_pAE",
        "i_pLDDT",
        "ss_pLDDT",
        "Unrelaxed_Clashes",
        "Relaxed_Clashes",
        "Binder_Energy_Score",
        "Surface_Hydrophobicity",
        "ShapeComplementarity",
        "PackStat",
        "dG",
        "dSASA",
        "dG/dSASA",
        "Interface_SASA_%",
        "Interface_Hydrophobicity",
        "n_InterfaceResidues",
        "n_InterfaceHbonds",
        "InterfaceHbondsPercentage",
        "n_InterfaceUnsatHbonds",
        "InterfaceUnsatHbondsPercentage",
        "Interface_Helix%",
        "Interface_BetaSheet%",
        "Interface_Loop%",
        "Binder_Helix%",
        "Binder_BetaSheet%",
        "Binder_Loop%",
        "InterfaceAAs",
        "Target_RMSD",
        "TrajectoryTime",
        "Notes",
        "TargetSettings",
        "Filters",
        "AdvancedSettings",
    ]

    # labels for mpnn designs
    core_labels = [
        "pLDDT",
        "pTM",
        "i_pTM",
        "pAE",
        "i_pAE",
        "i_pLDDT",
        "ss_pLDDT",
        "Unrelaxed_Clashes",
        "Relaxed_Clashes",
        "Binder_Energy_Score",
        "Surface_Hydrophobicity",
        "ShapeComplementarity",
        "PackStat",
        "dG",
        "dSASA",
        "dG/dSASA",
        "Interface_SASA_%",
        "Interface_Hydrophobicity",
        "n_InterfaceResidues",
        "n_InterfaceHbonds",
        "InterfaceHbondsPercentage",
        "n_InterfaceUnsatHbonds",
        "InterfaceUnsatHbondsPercentage",
        "Interface_Helix%",
        "Interface_BetaSheet%",
        "Interface_Loop%",
        "Binder_Helix%",
        "Binder_BetaSheet%",
        "Binder_Loop%",
        "InterfaceAAs",
        "Hotspot_RMSD",
        "Target_RMSD",
        "Binder_pLDDT",
        "Binder_pTM",
        "Binder_pAE",
        "Binder_RMSD",
    ]

    design_labels = [
        "Design",
        "Protocol",
        "Length",
        "Seed",
        "Helicity",
        "Target_Hotspot",
        "Sequence",
        "InterfaceResidues",
        "MPNN_score",
        "MPNN_seq_recovery",
    ]

    for label in core_labels:
        design_labels += ['Average_' + label] + [f'{i}_{label}' for i in range(1, 6)]

    design_labels += ['DesignTime', 'Notes', 'TargetSettings', 'Filters', 'AdvancedSettings']

    final_labels = ['Rank'] + design_labels

    return trajectory_labels, design_labels, final_labels

# Create base directions of the project
def generate_directories(design_path: str) -> Dict[str, str]:
    """Generate directory structure for design outputs."""
    design_path_names = [
        "Accepted",
        "Accepted/Ranked",
        "Accepted/Animation",
        "Accepted/Plots",
        "Accepted/Pickle",
        "Trajectory",
        "Trajectory/Relaxed",
        "Trajectory/Plots",
        "Trajectory/Clashing",
        "Trajectory/LowConfidence",
        "Trajectory/Animation",
        "Trajectory/Pickle",
        "MPNN",
        "MPNN/Binder",
        "MPNN/Sequences",
        "MPNN/Relaxed",
        "Rejected",
    ]
    design_paths = {}

    # make directories and set design_paths[FOLDER_NAME] variable
    for name in design_path_names:
        path = os.path.join(design_path, name)
        os.makedirs(path, exist_ok=True)
        design_paths[name] = path

    return design_paths

# generate CSV file for tracking designs not passing filters
def generate_filter_pass_csv(failure_csv: str, filter_json: str) -> None:
    """Generate CSV file for tracking filter passes."""
    if not os.path.exists(failure_csv):
        with open(filter_json, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Create a list of modified keys
        names = [
            "Trajectory_logits_pLDDT",
            "Trajectory_softmax_pLDDT",
            "Trajectory_one-hot_pLDDT",
            "Trajectory_final_pLDDT",
            "Trajectory_Contacts",
            "Trajectory_Clashes",
            "Trajectory_WrongHotspot",
        ]
        special_prefixes = ("Average_", "1_", "2_", "3_", "4_", "5_")
        tracked_filters = set()

        for key in data.keys():
            processed_name = key  # Use the full key by default

            # Check if the key starts with any special prefixes
            for prefix in special_prefixes:
                if key.startswith(prefix):
                    # Strip the prefix and use the remaining part
                    processed_name = key.split('_', 1)[1]
                    break

            # Handle 'InterfaceAAs' with appending amino acids
            if 'InterfaceAAs' in processed_name:
                # Generate 20 variations of 'InterfaceAAs' with amino acids appended
                amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
                for aa in amino_acids:
                    variant_name = f"InterfaceAAs_{aa}"
                    if variant_name not in tracked_filters:
                        names.append(variant_name)
                        tracked_filters.add(variant_name)
            elif processed_name not in tracked_filters:
                # Add processed name if it hasn't been added before
                names.append(processed_name)
                tracked_filters.add(processed_name)

        # make dataframe with 0s
        df = pd.DataFrame(columns=names)
        df.loc[0] = [0] * len(names)

        df.to_csv(failure_csv, index=False)

# update failure rates from trajectories and early predictions
def update_failures(failure_csv: str, failure_column_or_dict: Any) -> None:
    """Update failure tracking CSV with failed filter information."""
    failure_df = pd.read_csv(failure_csv)
    def strip_model_prefix(name: str) -> str:
        # Strips the model-specific prefix if it exists
        parts = name.split('_')
        if parts[0].isdigit():
            return '_'.join(parts[1:])
        return name
    # update dictionary coming from complex prediction
    if isinstance(failure_column_or_dict, dict):
        # Update using a dictionary of failures
        for filter_name, count in failure_column_or_dict.items():
            stripped_name = strip_model_prefix(filter_name)
            if stripped_name in failure_df.columns:
                failure_df[stripped_name] += count
            else:
                failure_df[stripped_name] = count
    else:
        # Update a single column from trajectory generation
        failure_column = strip_model_prefix(failure_column_or_dict)
        if failure_column in failure_df.columns:
            failure_df[failure_column] += 1
        else:
            failure_df[failure_column] = 1
    failure_df.to_csv(failure_csv, index=False)

# Check if number of trajectories generated
def check_n_trajectories(design_paths: Dict[str, str], advanced_settings: Dict[str, Any]) -> bool:
    """Check if required number of trajectories have been generated."""
    n_trajectories = [
        f
        for f in os.listdir(design_paths["Trajectory/Relaxed"])
        if f.endswith(".pdb") and not f.startswith(".")
    ]

    if (
        advanced_settings["max_trajectories"] is not False
        and len(n_trajectories) >= advanced_settings["max_trajectories"]
    ):
        print(
            f"Target number of {str(len(n_trajectories))} trajectories reached, "
            "stopping execution..."
        )
        return True
    return False


# Check if we have required number of accepted targets, rank them, and analyse
# sequence and structure properties
def check_accepted_designs(  # pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
    design_paths: Dict[str, str],
    mpnn_csv: str,
    final_labels: List[str],
    final_csv: str,
    advanced_settings: Dict[str, Any],
    target_settings: Dict[str, Any],
    design_labels: List[str],
) -> bool:
    """Check if required number of accepted designs have been generated."""
    accepted_binders = [
        f
        for f in os.listdir(design_paths["Accepted"])
        if (f.endswith(".pdb") and not f.startswith(".") and
            f.startswith(target_settings["binder_name"]))
    ]

    if len(accepted_binders) >= target_settings["number_of_final_designs"]:
        print(
            f"Target number {str(len(accepted_binders))} of designs reached! Reranking..."
        )

        # clear the Ranked folder in case we added new designs in the meantime so we rerank them all
        for f in os.listdir(design_paths["Accepted/Ranked"]):
            os.remove(os.path.join(design_paths["Accepted/Ranked"], f))

        # load dataframe of designed binders
        design_df = pd.read_csv(mpnn_csv)
        design_df = design_df.sort_values("Average_i_pTM", ascending=False)

        # create final csv dataframe to copy matched rows, initialize with the column labels
        final_df = pd.DataFrame(columns=final_labels)

        # check the ranking of the designs and copy them with new ranked IDs to the folder
        rank = 1
        for _, row in design_df.iterrows():
            for binder in accepted_binders:
                target_settings["binder_name"], model = binder.rsplit("_model", 1)
                if target_settings["binder_name"] == row["Design"]:
                    # rank and copy into ranked folder
                    row_data = {"Rank": rank, **{label: row[label] for label in design_labels}}
                    final_df = pd.concat(
                        [final_df, pd.DataFrame([row_data])], ignore_index=True
                    )
                    old_path = os.path.join(design_paths["Accepted"], binder)
                    new_path = os.path.join(
                        design_paths["Accepted/Ranked"],
                        f"{rank}_{target_settings['binder_name']}_model"
                        f"{model.rsplit('.', 1)[0]}.pdb",
                    )
                    shutil.copyfile(old_path, new_path)

                    rank += 1
                    break

        # save the final_df to final_csv
        final_df.to_csv(final_csv, index=False)

        # zip large folders to save space
        if advanced_settings["zip_animations"]:
            zip_and_empty_folder(design_paths["Trajectory/Animation"], '.html')

        if advanced_settings["zip_plots"]:
            zip_and_empty_folder(design_paths["Trajectory/Plots"], '.png')

        return True

    return False

# Load required helicity value
def load_helicity(advanced_settings: Dict[str, Any]) -> float:
    """Load helicity values from file."""
    if advanced_settings["random_helicity"] is True:
        # will sample a random bias towards helicity
        helicity_value = round(np.random.uniform(-3, 1),2)
    elif advanced_settings["weights_helicity"] != 0:
        # using a preset helicity bias
        helicity_value = advanced_settings["weights_helicity"]
    else:
        # no bias towards helicity
        helicity_value = 0
    return float(helicity_value)

# Report JAX-capable devices
def check_jax_gpu() -> None:
    """Check if JAX GPU is available."""
    devices = jax.devices()

    has_gpu = any(device.platform == "gpu" for device in devices)

    if not has_gpu:
        print("No GPU device found, terminating.")
        sys.exit(1)

    print("Available GPUs:")
    for i, device in enumerate(devices):
        print(f"{device.device_kind}{i + 1}: {device.platform}")


# check all input files being passed
def perform_input_check(args: Any) -> Tuple[str, str, str]:
    """Perform input validation checks."""
    # Get the directory of the current script
    binder_script_path = os.path.dirname(os.path.abspath(__file__))

    # Ensure settings file is provided
    if not args.settings:
        print("Error: --settings is required.")
        sys.exit(1)

    # Set default filters.json path if not provided
    if not args.filters:
        args.filters = os.path.join(
            binder_script_path, "settings_filters", "default_filters.json"
        )

    # Set a random advanced json settings file if not provided
    if not args.advanced:
        args.advanced = os.path.join(
            binder_script_path, "settings_advanced", "default_4stage_multimer.json"
        )

    return args.settings, args.filters, args.advanced


# check specific advanced settings
def perform_advanced_settings_check(advanced_settings: Dict[str, Any], bindstuff_folder: str) -> Dict[str, Any]:
    """Perform advanced settings validation checks."""
    # set paths to model weights and executables
    if bindstuff_folder == "colab":
        advanced_settings["af_params_dir"] = "/content/bindstuff/params/"
        advanced_settings["dssp_path"] = "/content/bindstuff/functions/dssp"
        advanced_settings["dalphaball_path"] = "/content/bindstuff/functions/DAlphaBall.gcc"
    else:
        # Set paths individually if they are not already set
        if not advanced_settings["af_params_dir"]:
            advanced_settings["af_params_dir"] = bindstuff_folder
        if not advanced_settings["dssp_path"]:
            advanced_settings["dssp_path"] = os.path.join(
                bindstuff_folder, "functions", "dssp"
            )
        if not advanced_settings["dalphaball_path"]:
            advanced_settings["dalphaball_path"] = os.path.join(
                bindstuff_folder, "functions", "DAlphaBall.gcc"
            )

    # check formatting of omit_AAs setting
    if advanced_settings["omit_AAs"] in [None, False, ""]:
        advanced_settings["omit_AAs"] = None
    elif isinstance(advanced_settings["omit_AAs"], str):
        advanced_settings["omit_AAs"] = advanced_settings["omit_AAs"].strip()

    return advanced_settings


# Load settings from JSONs
def load_json_settings(settings_json: str, filters_json: str, advanced_json: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load settings from JSON file."""
    # load settings from json files
    with open(settings_json, "r", encoding="utf-8") as file:
        target_settings = json.load(file)

    with open(advanced_json, "r", encoding="utf-8") as file:
        advanced_settings = json.load(file)

    with open(filters_json, "r", encoding="utf-8") as file:
        filters = json.load(file)

    return target_settings, advanced_settings, filters

# AF2 model settings, make sure non-overlapping models with template option are
# being used for design and re-prediction
def load_af2_models(af_multimer_setting: bool) -> Tuple[List[int], List[int], bool]:
    """Load AlphaFold2 model numbers from string."""
    if af_multimer_setting:
        design_models = [0,1,2,3,4]
        prediction_models = [0,1]
        multimer_validation = False
    else:
        design_models = [0,1]
        prediction_models = [0,1,2,3,4]
        multimer_validation = True

    return design_models, prediction_models, multimer_validation

# create csv for insertion of data
def create_dataframe(csv_file: str, columns: List[str]) -> None:
    """Create empty dataframe with specified labels."""
    if not os.path.exists(csv_file):
        df = pd.DataFrame(columns=columns)
        df.to_csv(csv_file, index=False)

# insert row of statistics into csv
def insert_data(csv_file: str, data_array: List[Any]) -> None:
    """Insert data row into dataframe."""
    df = pd.DataFrame([data_array])
    df.to_csv(csv_file, mode='a', header=False, index=False)

# save generated sequence
def save_fasta(design_name: str, sequence: str, design_paths: Dict[str, str]) -> None:
    """Save sequence to FASTA file."""
    fasta_path = os.path.join(design_paths["MPNN/Sequences"], design_name+".fasta")
    with open(fasta_path,"w", encoding='utf-8') as fasta:
        line = f'>{design_name}\n{sequence}'
        fasta.write(line+"\n")

# clean unnecessary rosetta information from PDB
def clean_pdb(pdb_file: str) -> None:
    """Clean PDB file by removing unwanted lines."""
    # Read the pdb file and filter relevant lines
    with open(pdb_file, 'r', encoding='utf-8') as f_in:
        relevant_lines = [line for line in f_in
                         if line.startswith(('ATOM', 'HETATM', 'MODEL',
                                           'TER', 'END', 'LINK'))]

    # Write the cleaned lines back to the original pdb file
    with open(pdb_file, 'w', encoding='utf-8') as f_out:
        f_out.writelines(relevant_lines)

def zip_and_empty_folder(folder_path: str, extension: str) -> None:
    """Zip folder contents and empty the folder."""
    folder_basename = os.path.basename(folder_path)
    zip_filename = os.path.join(os.path.dirname(folder_path),
                               folder_basename + ".zip")

    # Open the zip file in 'a' mode to append if it exists, otherwise create a new one
    with zipfile.ZipFile(zip_filename, "a", zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(folder_path):
            if file.endswith(extension):
                # Create an absolute path
                file_path = os.path.join(folder_path, file)
                # Add file to zip file, replacing it if it already exists
                zipf.write(file_path, arcname=file)
                # Remove the file after adding it to the zip
                os.remove(file_path)
    print(f"Files in folder '{folder_path}' have been zipped and removed.")

# calculate averages for statistics
def calculate_averages(statistics: Dict[int, Dict[str, Any]], handle_aa: bool = False) -> Dict[str, Any]:  # pylint: disable=too-many-nested-blocks
    """Calculate averages for dataframe columns."""
    # Initialize a dictionary to hold the sums of each statistic
    sums = {}
    # Initialize a dictionary to hold the sums of each amino acid count
    aa_sums = {}

    # Iterate over the model numbers
    for model_num in range(1, 6):  # assumes models are numbered 1 through 5  # pylint: disable=too-many-nested-blocks
        # Check if the model's data exists
        if model_num in statistics:
            # Get the model's statistics
            model_stats = statistics[model_num]
            # For each statistic, add its value to the sum
            for stat, value in model_stats.items():
                # If this is the first time we've seen this statistic, initialize its sum to 0
                if stat not in sums:
                    sums[stat] = 0

                if value is None:
                    value = 0

                # If the statistic is mpnn_interface_AA and we're supposed to handle it
                # separately, do so
                if (handle_aa and stat == 'InterfaceAAs' and
                    isinstance(value, dict)):
                    for aa, count in value.items():
                        # If this is the first time we've seen this amino acid,
                        # initialize its sum to 0
                        if aa not in aa_sums:
                            aa_sums[aa] = 0
                        aa_sums[aa] += count
                else:
                    sums[stat] += value

    # Now that we have the sums, we can calculate the averages
    averages = {stat: round(total / len(statistics), 2) for stat, total in sums.items()}

    # If we're handling aa counts, calculate their averages
    if handle_aa:
        aa_averages = {aa: round(total / len(statistics), 2)
                      for aa, total in aa_sums.items()}
        averages['InterfaceAAs'] = float(sum(aa_averages.values()) / len(aa_averages)) if aa_averages else 0.0

    return averages

# filter designs based on feature thresholds
def check_filters(mpnn_data: List[Any], design_labels: List[str], filters: Dict[str, Any]) -> Any:  # pylint: disable=too-many-branches
    """Check if design passes all specified filters."""
    # check mpnn_data against labels
    mpnn_dict = dict(zip(design_labels, mpnn_data))

    unmet_conditions = []

    # check filters against thresholds
    for label, conditions in filters.items():
        # special conditions for interface amino acid counts
        if label in (
            "Average_InterfaceAAs",
            "1_InterfaceAAs",
            "2_InterfaceAAs",
            "3_InterfaceAAs",
            "4_InterfaceAAs",
            "5_InterfaceAAs",
        ):
            for aa, aa_conditions in conditions.items():
                label_dict = mpnn_dict.get(label)
                if label_dict is None:
                    continue
                value = label_dict.get(aa)
                if value is None or aa_conditions["threshold"] is None:
                    continue
                if aa_conditions["higher"]:
                    if value < aa_conditions["threshold"]:
                        unmet_conditions.append(f"{label}_{aa}")
                else:
                    if value > aa_conditions["threshold"]:
                        unmet_conditions.append(f"{label}_{aa}")
        else:
            # if no threshold, then skip
            value = mpnn_dict.get(label)
            if value is None or conditions["threshold"] is None:
                continue
            if conditions["higher"]:
                if value < conditions["threshold"]:
                    unmet_conditions.append(label)
            else:
                if value > conditions["threshold"]:
                    unmet_conditions.append(label)

    # if all filters are passed then return True
    if len(unmet_conditions) == 0:
        return True
    return unmet_conditions

def create_target_settings_from_form(design_path, binder_name, starting_pdb, chains, target_hotspot_residues, lengths, number_of_final_designs, load_previous_target_settings):
    """Creates a target settings dictionary from form inputs."""
    if load_previous_target_settings:
        return load_previous_target_settings

    lengths = [int(x.strip()) for x in lengths.split(',') if len(lengths.split(',')) == 2]
    if len(lengths) != 2:
        raise ValueError("Incorrect specification of binder lengths.")

    settings = {
        "design_path": design_path,
        "binder_name": binder_name,
        "starting_pdb": starting_pdb,
        "chains": chains,
        "target_hotspot_residues": target_hotspot_residues,
        "lengths": lengths,
        "number_of_final_designs": number_of_final_designs
    }

    target_settings_path = os.path.join(design_path, binder_name + ".json")
    os.makedirs(design_path, exist_ok=True)

    with open(target_settings_path, 'w') as f:
        json.dump(settings, f, indent=4)

    return target_settings_path

def get_advanced_settings_path_from_form(design_protocol, interface_protocol, template_protocol, prediction_protocol):
    """Gets the advanced settings path from form inputs."""
    if design_protocol == "Default":
        design_protocol_tag = "default_4stage_multimer"
    elif design_protocol == "Beta-sheet":
        design_protocol_tag = "betasheet_4stage_multimer"
    elif design_protocol == "Peptide":
        design_protocol_tag = "peptide_3stage_multimer"
    else:
        raise ValueError("Unsupported design protocol")

    if interface_protocol == "AlphaFold2":
        interface_protocol_tag = ""
    elif interface_protocol == "MPNN":
        interface_protocol_tag = "_mpnn"
    else:
        raise ValueError("Unsupported interface protocol")

    if template_protocol == "Default":
        template_protocol_tag = ""
    elif template_protocol == "Masked":
        template_protocol_tag = "_flexible"
    else:
        raise ValueError("Unsupported template protocol")

    if design_protocol in ["Peptide"]:
        prediction_protocol_tag = ""
    else:
        if prediction_protocol == "Default":
            prediction_protocol_tag = ""
        elif prediction_protocol == "HardTarget":
            prediction_protocol_tag = "_hardtarget"
        else:
            raise ValueError("Unsupported prediction protocol")

    return "/content/bindstuff/settings_advanced/" + design_protocol_tag + interface_protocol_tag + template_protocol_tag + prediction_protocol_tag + ".json"

def get_filter_settings_path_from_form(filter_option):
    """Gets the filter settings path from form inputs."""
    if filter_option == "Default":
        return "/content/bindstuff/settings_filters/default_filters.json"
    elif filter_option == "Peptide":
        return "/content/bindstuff/settings_filters/peptide_filters.json"
    elif filter_option == "Relaxed":
        return "/content/bindstuff/settings_filters/relaxed_filters.json"
    elif filter_option == "Peptide_Relaxed":
        return "/content/bindstuff/settings_filters/peptide_relaxed_filters.json"
    elif filter_option == "None":
        return "/content/bindstuff/settings_filters/no_filters.json"
    else:
        raise ValueError("Unsupported filter type")
