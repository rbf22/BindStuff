import argparse
import os
from functions.generic_utils import (
    check_jax_gpu,
    perform_input_check,
    load_json_settings,
    load_af2_models,
    perform_advanced_settings_check,
    generate_directories,
    generate_dataframe_labels,
    create_dataframe,
    generate_filter_pass_csv,
)
from functions.pipeline import run_pipeline

def main():
    """Main function to run the binder design pipeline."""
    # Check if JAX-capable GPU is available, otherwise exit
    check_jax_gpu()

    ######################################
    ### parse input paths
    parser = argparse.ArgumentParser(description='Script to run BindStuff binder design.')

    parser.add_argument('--settings', '-s', type=str, required=True,
                        help='Path to the basic settings.json file. Required.')
    parser.add_argument('--filters', '-f', type=str, default='./settings_filters/default_filters.json',
                        help='Path to the filters.json file used to filter design. If not provided, default will be used.')
    parser.add_argument('--advanced', '-a', type=str, default='./settings_advanced/default_4stage_multimer.json',
                        help='Path to the advanced.json file with additional design settings. If not provided, default will be used.')

    args = parser.parse_args()

    # perform checks of input setting files
    settings_path, filters_path, advanced_path = perform_input_check(args)

    ### load settings from JSON
    target_settings, advanced_settings, filters = load_json_settings(settings_path, filters_path, advanced_path)

    settings_file = os.path.basename(settings_path).split('.')[0]
    filters_file = os.path.basename(filters_path).split('.')[0]
    advanced_file = os.path.basename(advanced_path).split('.')[0]

    ### load AF2 model settings
    design_models, prediction_models, multimer_validation = load_af2_models(advanced_settings["use_multimer_design"])

    ### perform checks on advanced_settings
    bindstuff_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    advanced_settings = perform_advanced_settings_check(advanced_settings, bindstuff_folder)

    ### generate directories, design path names can be found within the function
    design_paths = generate_directories(target_settings["design_path"])

    ### generate dataframes
    trajectory_labels, design_labels, final_labels = generate_dataframe_labels()

    trajectory_csv = os.path.join(target_settings["design_path"], 'trajectory_stats.csv')
    mpnn_csv = os.path.join(target_settings["design_path"], 'mpnn_design_stats.csv')
    final_csv = os.path.join(target_settings["design_path"], 'final_design_stats.csv')
    failure_csv = os.path.join(target_settings["design_path"], 'failure_csv.csv')

    create_dataframe(trajectory_csv, trajectory_labels)
    create_dataframe(mpnn_csv, design_labels)
    create_dataframe(final_csv, final_labels)
    generate_filter_pass_csv(failure_csv, args.filters)

    run_pipeline(
        target_settings=target_settings,
        advanced_settings=advanced_settings,
        filters=filters,
        design_models=design_models,
        prediction_models=prediction_models,
        multimer_validation=multimer_validation,
        design_paths=design_paths,
        trajectory_csv=trajectory_csv,
        mpnn_csv=mpnn_csv,
        final_csv=final_csv,
        failure_csv=failure_csv,
        settings_file=settings_file,
        filters_file=filters_file,
        advanced_file=advanced_file,
        trajectory_labels=trajectory_labels,
        design_labels=design_labels,
        final_labels=final_labels
    )

if __name__ == '__main__':
    main()
