function ground_truth = estimate_lags_mediaeval(annotation_csv_path, features_csv_path, subj_id_regexp, output_folder)
    start_time = 15.0; % Mediaeval data starts samples at 15 seconds into each song
    frequency = 2; % Mediaeval data is sampled at 2Hz
    [annotations, label_sequences] = import_mediaeval_annotation(annotation_csv_path, subj_id_regexp);
    if isempty(annotations)
        return;
    end

    addpath(genpath([cd '/mariooryad_lag_estimation']))
    feature_sequences = get_features_mediaeval(features_csv_path);
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
    
    %output_folder = '../../results/datasets/mediaeval/arousal'
    header = {'Time_seconds', 'Data'};    
    times = (1.0/frequency)*(0:(size(mariooryad_shifted_labels,2)-1)) + start_time;
    for ann_idx = 1:size(mariooryad_shifted_labels,1)
        [file_path, task_name, task_ext] = fileparts(annotation_csv_path);
        output_file_name = strcat(task_name, '_ann', int2str(ann_idx), '.csv');
        output_file_path = strcat(output_folder, '/', output_file_name);
        out_mat = [times; mariooryad_shifted_labels(ann_idx,:)]';
        write_csv_file(output_file_path, out_mat, header);
    end

    ground_truth = [times; mean(mariooryad_shifted_labels,1)]';
end
