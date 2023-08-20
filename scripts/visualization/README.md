# Visualizing results
**Plotting fused TSS with annotations:** Running `fused_tss_and_annotations.py` performs this task, e.g.):
<blockquote>
python fused_tss_and_annotations.py --annotations_path ../../results/main_study/annotations/cleaned_selected_aligned --fused_tss_path ../../results/main_study/tss/Good_Boys_cut2/fused_segment_sequence.csv --file_name_filter_str=Good_Boys_cut2
</blockquote>

**Plotting raw PAGAN annotations:** The `plot_pagan_logs.py` file visualizes PAGAN annotations (also called logs):
<blockquote>
python plot_pagan_logs.py --input_log "../../results/main_study/annotations/cleaned/Movie-Violence-Rating-Experiment-.csv"
</blockquote>
