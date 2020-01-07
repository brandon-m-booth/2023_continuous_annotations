function feature_sequences = get_features_mediaeval(features_csv_path)
    [features_file_path,features_file_name,features_file_ext] = fileparts(features_csv_path);
    feature_sequence_mat = [];
    features_files = dir(features_file_path);
    for file_index=1:length(features_files)
        file_name = features_files(file_index).name;
        
        % Ignore files not requested by user input
        do_skip_file = ~strcmp(file_name, strcat(features_file_name, features_file_ext));
        if do_skip_file
            continue;
        end
        
        csv_data = read_csv_file(fullfile(features_file_path, file_name), ';');
        csv_data = cast_to_best_data_type(csv_data(2:end,2:end)); % Discard the header and workerID column
        feature_sequence_mat = [feature_sequence_mat, csv_data];
    end
    feature_sequences = cell(0);
    feature_sequences{1} = feature_sequence_mat;
end
