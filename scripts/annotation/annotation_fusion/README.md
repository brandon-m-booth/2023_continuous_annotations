# Annotation Fusion
Two annotation fusion methods are provided: an align-and-average fusion (the paper's baseline method) and ordinal deformation fusion (the proposed method).  Users wanting to run the baseline or proposed methods can provide their own annotation files according to the following:
1. All annotations are in one folder and csv-formatted
1. Annotation files contain the experiment name (or any string, e.g., "Experiment1") and some unique identifier for the annotator, e.g.: `<experiment-name>_opt_annotations_<annotator-id>.csv`
1. Each file contains a single annotation from one annotator with two columns: the first called "Time(sec)" containing the elapsed time in seconds since the beginning of the experiment, and the second named using the annotator's ID (e.g., "AF8GOPFSTWCH8") and containing the value assigned by that annotator in each time frame.

## Average Fusion (Baseline method)
Code for averaging is contained in the `average` folder.  The `baseline_fusion.py` script is designed to work on the annotations in the movie violence case study, but a simple average of the annotation frames across the cleaned, selected, and aligned annotations yields the same result. To run the code on the movie violence study, we ran the following command (note that `--show_plots` is optional):
<blockquote>
python baseline_fusion.py --input_clip_annotations_path ../../../../results/main_study/annotations/cleaned_selected_aligned --clip_times_csv_path ../../../../dataset/movie_violence/movie_cut_times.csv --movie_name_regex "(.*?)_cut.*\.csv" --clip_number_regex ".*_cut(\d+)_opt.*\.csv" --output_path .../../../../results/main_study/baseline --show_plots
</blockquote>

## Ordinal Deformation (Proposed method)
Code for the proposed method is broken into steps and contained in the `ordinal_deformation` folder.  The steps are summarized below with example commands used for the movie violence study:
1. From `1_tsr`, run trapezoidal segmented regression for a single input annotation over a range of numbers of segments: 
<blockquote>
python compute_tsr.py --input ../../../../results/main_study/annotations/cleaned_selected_aligned/Good_Boys_cut1_opt_annotations_AF8GOPFSTWCH8.csv --segments 8-15 --output ../../../../results/main_study/tsr/Good_Boys_cut1/Good_Boys_cut1_opt_annotations_AF8GOPFSTWCH8.csv
</blockquote>
This should be repeated for each annotation for each experiment (e.g., cut1, cut2, ...).  The output is a time series with the same format as the input but only the vertices of the trapezoidal signal are specified (i.e., the full signal is made by "connecting the dots")

2. From `2_optimize_tsr`, find the optimal number of segments for a given set of proposed trapezoidal segmented regressions for an annotation (note `--show_plots` is optional):
<blockquote>
python optimize_tsr.py --input_signal_path ../../../../../results/main_study/annotations/cleaned_selected_aligned --input_tsr_path ../../../../../results/main_study/tsr/Good_Boys_cut1 --output_path ../../../../../results/main_study/tsr_opt/Good_Boys_cut1 --subj_id_regex "annotations\_(.*?)\.csv" --subj_id_tsr_regex "annotations\_(.*?)\_T\d+\.csv" --segment_num_regex "\_T(\d+)\.csv" --file_name_filter_str="Good_Boys_cut1" --show_plots
</blockquote>
This should be repeated for each annotation for each experiment.  The output is a set of the best TSR files, independently decided per annotation, according to the heuristic described in the paper.

**Note:** The `extract_const_intervals.py` script provides helper functions to extract the constant intervals from each TSR.  The `optimize_tsr.py` function uses these helpers, so this file does not need to be run separately, but it can be used to pull the constant regions out from a TSR file.
 
3. From `3_trapezoidal_segment_fusion`, fuse the TSR annotation approximations per experiment: 
<blockquote>
python trapezoidal_fusion.py --input ../../../../../results/main_study/tsr_opt/Good_Boys_cut1 --output_path ../../../../../results/main_study/tss/Good_Boys_cut1
</blockquote>

 Two files are output: 1) `constant_interval_indices.csv` containing two columns with each row dictating the start and end indices of each constant segment based on the TSR regressions, and 2) `fused_segment_sequence.csv` containing the trapezoidal segment sequence (see the paper) of the fused TSR signals.
 
4. From `4_generate_mturk_pairings`, there are scripts to generate a list of pairings from which to crowd source ordinal comparisons and also scripts to process the results from crowd sourcing:
    4.1. For example, for generating pairs for the movie violence case study: 
    
    <blockquote>
    python gen_pairs_mturk_movie_violence.py --config_json sample_gen_mturk_movie_violence_config.json
    </blockquote>
    
	This script takes a configuration file as input, which provides a list of file inputs for each experiment, and outputs a csv file for each relevant comparison (e.g., pairs or triplets) with start and end times for each stimulus (e.g., movie clip) when the stimulus is approximately constant.  The sample config provides a starting point for running this script for the movie violence study: `python gen_pairs_mturk_movie_violence.py --config_json`.  The json format is as follows:

   * `output_path`: the output folder location
   * `base_url`: a string prepended to the new stimulus path for each output row
   * `lag_shift_seconds`: a globally corrective lag shift in seconds to fixup systemic misalignments due to network latency
   * `clip_data`: A list containing dictionary elements with the following keys: "const_intervals_path" (path to an individual "constant_interval_indicies.csv" file), "fused_seg_seq_path" (path to a fused TSR file for a given experiment), "source_mp4_path" (path to the source stimulus or mp4/video file)

    4.2. To convert the Mechanical Turk (crowd sourcing) pairwise or triplet comparison results back into an ordered list of ordinal preferences compatible with ordinal triplet embedding, run:
    
    <blockquote>
    python generate_pairs_matrix_from_mturk.py --mturk_batch_path ../../../../../results/main_study/mturk_batch_results/pairs --output_path ../../../../../results/main_study/mturk_batch_results/pairs
    </blockquote> 

5. From `5_ordinate_constant_segments`, run a stochastic pairwise or triplet embedding solver to generate a signal from the ordered list of pairs/triplets from crowd sourcing. Following the tSTE library, this code is written for Matlab; for example, for the movie violence study:

<blockquote>
tspeEmbedding("../../../../../results/main_study/mturk_batch_results/pairs/mturk_batch_results_pairs.csv", "../../../../../results/main_study/embedding/pairs/embedding.csv")
</blockquote>

6. From `6_construct_signal`, combine the ordinal embedding results and the fused TSRs to generate a new and more accurate signal.  For the movie violence study, we used this command  (note that showing plots is optional):

<blockquote>
python construct_signals_movie_violence.py --json_config_path ../config/movie_violence_study_config.json --show_plots
</blockquote>

Like the previous step involving json config files as input, this script takes path to specific files as inputs (see the `movie_violence_study_config.json` sample).  Other options include:

   * `flip_embedding`: This flips the resulting signal so high values are low and vice versa (because ordinal embedding solvers cannot do this automatically)
   * `norm_scale`: Lists the low and high extreme values in the annotation scale (e.g., [-100,100])
   * `interpolation_method`: accepts either "average" or "trapezoidal".  If "average", then the annotations from each annotator are averaged and spliced into the final signal to fill in between the constant regions.  If "trapezoidal", then a line is used to connect the constant regions in the final deformed signal.
 
**Note**: `FileIO.py` and `PrettyPlotter.py` are helper libraries and are not meant to run standalone.
