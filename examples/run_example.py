import sys
import os
import click
import shutil
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
    is_gpu_available,
)
from functions.pipeline import run_pipeline


@click.command()
@click.option('--run-on-cpu', is_flag=True, help='Run the pipeline on CPU.')
def main(run_on_cpu):
    """Main function to run the binder design pipeline."""
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

    target_settings_path = create_target_settings_from_form(
        design_path, binder_name, starting_pdb, chains, target_hotspot_residues,
        lengths, number_of_final_designs, load_previous_target_settings
    )

    print(f"Binder design settings updated at: {currenttime}")
    print(f"New .json file with target settings has been generated in: {target_settings_path}")

    design_protocol = "Default"
    prediction_protocol = "Default"
    interface_protocol = "AlphaFold2"
    template_protocol = "Default"

    advanced_settings_path = get_advanced_settings_path_from_form(
        design_protocol, interface_protocol, template_protocol, prediction_protocol
    )

    print(f"Advanced design settings updated at: {currenttime}")
    print(f"New .json file with target settings has been generated in: {advanced_settings_path}")

    filter_option = "Default"
    filter_settings_path = get_filter_settings_path_from_form(filter_option)

    print(f"Filter settings updated at: {currenttime}")
    print(f"New .json file with target settings has been generated in: {filter_settings_path}")

    target_settings, advanced_settings, filters = load_json_settings(
        target_settings_path, filter_settings_path, advanced_settings_path
    )

    settings_file = os.path.basename(target_settings_path).split('.')[0]
    filters_file = os.path.basename(filter_settings_path).split('.')[0]
    advanced_file = os.path.basename(advanced_settings_path).split('.')[0]

    design_models, prediction_models, multimer_validation = load_af2_models(
        advanced_settings["use_multimer_design"]
    )

    bindstuff_folder = "."
    advanced_settings = perform_advanced_settings_check(advanced_settings, bindstuff_folder)

    design_paths = generate_directories(target_settings["design_path"])

    # Copy settings files to the design directory
    shutil.copy(advanced_settings_path, target_settings["design_path"])
    shutil.copy(filter_settings_path, target_settings["design_path"])

    trajectory_labels, design_labels, final_labels = generate_dataframe_labels()

    trajectory_csv = os.path.join(target_settings["design_path"], 'trajectory_stats.csv')
    mpnn_csv = os.path.join(target_settings["design_path"], 'mpnn_design_stats.csv')
    final_csv = os.path.join(target_settings["design_path"], 'final_design_stats.csv')
    failure_csv = os.path.join(target_settings["design_path"], 'failure_csv.csv')

    create_dataframe(trajectory_csv, trajectory_labels)
    create_dataframe(mpnn_csv, design_labels)
    create_dataframe(final_csv, final_labels)
    generate_filter_pass_csv(failure_csv, filter_settings_path)

    run_pipeline_args = {
        "target_settings": target_settings,
        "advanced_settings": advanced_settings,
        "filters": filters,
        "design_models": design_models,
        "prediction_models": prediction_models,
        "multimer_validation": multimer_validation,
        "design_paths": design_paths,
        "trajectory_csv": trajectory_csv,
        "mpnn_csv": mpnn_csv,
        "final_csv": final_csv,
        "failure_csv": failure_csv,
        "settings_file": settings_file,
        "filters_file": filters_file,
        "advanced_file": advanced_file,
        "trajectory_labels": trajectory_labels,
        "design_labels": design_labels,
        "final_labels": final_labels
    }

    if run_on_cpu:
        print("\n\n##################################################################")
        print("WARNING: Running on CPU. This will be extremely slow.")
        print("##################################################################\n\n")
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
        run_pipeline(**run_pipeline_args)
    elif is_gpu_available():
        print("\n\n##################################################################")
        print("GPU detected. Running pipeline on GPU.")
        print("##################################################################\n\n")
        run_pipeline(**run_pipeline_args)
    else:
        print("\n\n##################################################################")
        print("Script setup complete. Ready to run pipeline.")
        print("NOTE: This is a dry run. No GPU was detected.")
        print("To run on CPU, use the --run-on-cpu flag.")
        print("##################################################################\n\n")
        sys.exit(0)


if __name__ == '__main__':
    main()
