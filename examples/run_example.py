import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime
from functions.generic_utils import (
    create_target_settings_from_form,
    get_advanced_settings_path_from_form,
    get_filter_settings_path_from_form,
    load_json_settings,
    load_af2_models,
    perform_advanced_settings_check,
    generate_directories,
    generate_dataframe_labels,
    create_dataframe,
    generate_filter_pass_csv,
)

currenttime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

design_path = "examples/designs"

os.makedirs(design_path, exist_ok=True)
print("BindStuff folder successfully created in your drive!")

binder_name = "PDL1"

starting_pdb = "examples/PDL1.pdb"

chains = "A"

target_hotspot_residues = ""

lengths = "70,150"

number_of_final_designs = 100

load_previous_target_settings = ""

target_settings_path = create_target_settings_from_form(design_path, binder_name, starting_pdb, chains, target_hotspot_residues, lengths, number_of_final_designs, load_previous_target_settings)

currenttime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"Binder design settings updated at: {currenttime}")
print(f"New .json file with target settings has been generated in: {target_settings_path}")

design_protocol = "Default" 

prediction_protocol = "Default"

interface_protocol = "AlphaFold2"

template_protocol = "Default"

advanced_settings_path = get_advanced_settings_path_from_form(design_protocol, interface_protocol, template_protocol, prediction_protocol)

currenttime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Advanced design settings updated at: {currenttime}")
print(f"New .json file with target settings has been generated in: {advanced_settings_path}")

filter_option = "Default"

filter_settings_path = get_filter_settings_path_from_form(filter_option)

currenttime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Filter settings updated at: {currenttime}")
print(f"New .json file with target settings has been generated in: {filter_settings_path}")

# check_jax_gpu()

target_settings, advanced_settings, filters = load_json_settings(target_settings_path, filter_settings_path, advanced_settings_path)

settings_file = os.path.basename(target_settings_path).split('.')[0]
filters_file = os.path.basename(filter_settings_path).split('.')[0]
advanced_file = os.path.basename(advanced_settings_path).split('.')[0]

design_models, prediction_models, multimer_validation = load_af2_models(advanced_settings["use_multimer_design"])

bindstuff_folder = "."
advanced_settings = perform_advanced_settings_check(advanced_settings, bindstuff_folder)

design_paths = generate_directories(target_settings["design_path"])

trajectory_labels, design_labels, final_labels = generate_dataframe_labels()

trajectory_csv = os.path.join(target_settings["design_path"], 'trajectory_stats.csv')
mpnn_csv = os.path.join(target_settings["design_path"], 'mpnn_design_stats.csv')
final_csv = os.path.join(target_settings["design_path"], 'final_design_stats.csv')
failure_csv = os.path.join(target_settings["design_path"], 'failure_csv.csv')

create_dataframe(trajectory_csv, trajectory_labels)
create_dataframe(mpnn_csv, design_labels)
create_dataframe(final_csv, final_labels)
generate_filter_pass_csv(failure_csv, filter_settings_path)

# run_pipeline(
#     target_settings=target_settings,
#     advanced_settings=advanced_settings,
#     filters=filters,
#     design_models=design_models,
#     prediction_models=prediction_models,
#     multimer_validation=multimer_validation,
#     design_paths=design_paths,
#     trajectory_csv=trajectory_csv,
#     mpnn_csv=mpnn_csv,
#     final_csv=final_csv,
#     failure_csv=failure_csv,
#     settings_file=settings_file,
#     filters_file=filters_file,
#     advanced_file=advanced_file,
#     trajectory_labels=trajectory_labels,
#     design_labels=design_labels,
#     final_labels=final_labels
# )

print("\n\n##################################################################")
print("Script setup complete. Ready to run pipeline.")
print("NOTE: This is a dry run. The actual pipeline is not being run because the model parameters are not downloaded and no GPU is available.")
print("To run the full pipeline, run `make download-params` and run on a machine with a GPU.")
print("##################################################################\n\n")
