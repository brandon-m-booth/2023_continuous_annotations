function ground_truth = estimate_lags(task_name, frequency, output_folder)
    [annotations, label_sequences] = import_annotations(task_name, frequency);
    if isempty(annotations)
        return;
    end

    addpath(genpath([cd '/mariooryad_lag_estimation']))
    feature_sequences = get_features(task_name, frequency);
    max_lag_frames = 10*frequency;
    mariooryad_lags = estimate_lags_mariooryad(annotations, feature_sequences, label_sequences, max_lag_frames);
    shifted_labels = cell(1,length(label_sequences));
    min_length = inf;
    for label_seq_idx = 1:length(label_sequences)
        lag = mariooryad_lags.annotator_lags{label_seq_idx}(1);
        shifted_labels{label_seq_idx} = label_sequences{label_seq_idx}(lag+1:end);
        if length(shifted_labels{label_seq_idx}) < min_length
            min_length = length(shifted_labels{label_seq_idx});
        end
    end

    % Make all label sequences as long as the shortest one
    mariooryad_shifted_labels = zeros(length(label_sequences), min_length);
    for label_seq_idx = 1:length(label_sequences)
        shifted_labels{label_seq_idx} = shifted_labels{label_seq_idx}(1:min_length);
        mariooryad_shifted_labels(label_seq_idx,:) = shifted_labels{label_seq_idx};
    end
    
    %output_folder = '../../results/datasets/green_intensity/TaskA/annotations_1hz_aligned'
    header = {'Time_seconds', 'Data'};    
    times = (1.0/frequency)*(0:(size(mariooryad_shifted_labels,2)-1));
    for ann_idx = 1:size(mariooryad_shifted_labels,1)
        output_file_name = strcat(task_name, '_ann', int2str(ann_idx), '.csv');
        output_file_path = strcat(output_folder, '/', output_file_name);
        out_mat = [times; mariooryad_shifted_labels(ann_idx,:)]';
        write_csv_file(output_file_path, out_mat, header);
    end

    ground_truth = [times; mean(mariooryad_shifted_labels,1)]';
end
