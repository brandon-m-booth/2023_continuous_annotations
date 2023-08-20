# Analyzing Results

**Comparing agreement between annotation batches:** Running `compare_annotation_batches.py` measures the agreement between annotation batches within the same experiment. For the movie violence study, we ran:
<blockquote>
python compare_annotation_batches.py --json_config config/movie_violence_study_annotation_batches.json
</blockquote>

**Comparing annotation agreement across experiments:** Comparison of the average (baseline) and ordinal deformation (proposed) methods' agreement with movie ratings is performed by `compare_annotation_ratings.py`.  This code requires a pickle file with all CSM ratings, which is not provided in this repository due to licensing, but the code is here for completeness:
<blockquote>
python compare_annotation_ratings.py --baseline_annotations ../../results/main_study/baseline --warped_annotations_path ../../results/main_study/warped_annotations/pairs/pairs_merged --csm_ratings_path ../../dataset/movie_violence/main_study/csm_ratings/commonsense.pkl --output_folder ./../../figures/tex
</blockquote>
