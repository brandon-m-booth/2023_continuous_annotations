function feature_sequences = get_features(task_name, frequency)
    [pathstr, filename, ext] = fileparts(mfilename('fullpath'));
    features_file_exp = strcat(pathstr,'/../../datasets/green_intensity/',task_name,'/features_',num2str(frequency),'hz/*.csv');
    [features_file_path,name,ext] = fileparts(features_file_exp);
    feature_sequence_mat = [];
    features_files = dir(features_file_exp);
    for file_index=1:length(features_files)
        file_name = features_files(file_index).name;
        csv_data = read_csv_file(fullfile(features_file_path, file_name), ',');
        csv_data = cast_to_best_data_type(csv_data(2:end,2:end)); % Discard the header and time column
        feature_sequence_mat = [feature_sequence_mat, csv_data];
    end
    feature_sequences = cell(0);
    feature_sequences{1} = feature_sequence_mat;
end
