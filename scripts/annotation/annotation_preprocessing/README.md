# Cleaning PAGAN Annotations
The PAGAN tool streams annotation data in real time over the web to a database for storage.  Since packet losses, connectivity issues, and out-of-order delivery can affect the resulting annotations, we preprocess the annotations to clean these artifacts. The `clean_pagan_logs.py` file does two things: 1) fills in frames with NaN values where "skips" occur in the time series (i.e., missing packets of data), 2) performs a sort to reorder annotation data received out of order. The following command cleans the main study annotations:
<blockquote>
python clean_pagan_logs.py --input_log ../../../dataset/movie_violence/main_study/annotations --output_log ../../../results/main_study/annotations/cleaned
</blockquote>

There is an optional flag `--show_plots` to display the annotations collected in each experiment log.

# Aligning and Selecting PAGAN Annotations
The `align_and_select_pagan.py` script aligns and selects annotations as described in the paper.  Namely, each pair of annotations in an experiment is aligned then an agreement measure is used on each pair and tabulated in a similarity matrix.  Then a binary clustering technique uses this similarity matrix to separate the inliers from outliers.  The script takes PAGAN logs as input and outputs a `*_clusters.csv` and `*_opt_annotation.csv` file. The clusters file shows which annotations are considered inliers or outliers based on different alignment and clustering methods (e.g., using Cohen's kappa similarity with agglomerative clustering or pairwise dynamic time warping [DTW] with signed differential agreement [SDA] spectral clustering).  The optimal annotations file outputs the times series annotations for each inlier according to the binary spectral clustering of pairwise DTW aligned and compared using SDA. To process the main case study, we used this command:
<blockquote>
python align_and_select_pagan.py --input_log ../../../results/main_study/annotations/cleaned --output_path ../../../results/main_study/agreement_filtering
</blockquote>

# Aligning Selected Annotations
Once the inlier annotations are selected, these annotations are aligned with respect to each other and excluding the outliers.  The `align_selected_dtw.py` script performs this task using dynamic time warping.  For the main movie violence study, the following command performs this task (note that `--show_plots` is optional):
<blockquote>
python align_selected_dtw.py --input_annos ../../../results/main_study/agreement_filtering/ --output_path ../../../results/annotations/clean_selected_aligned --max_warp_seconds 5 --resample_hz 1 --show_plots
</blockquote>

The result is a folder containing csv files, one for each annotation, with two columns: the first called "Time(sec)" containing the elapsed time since the beginning of the video, and the second named after the annotator ID (e.g., "AF8GOPFSTWCH8") containing the value assigned by that annotator at each time frame. The file names contain information about the experiment and the annotator: `<experiment-name>_opt_annotations_<annotator-id>.csv`.
