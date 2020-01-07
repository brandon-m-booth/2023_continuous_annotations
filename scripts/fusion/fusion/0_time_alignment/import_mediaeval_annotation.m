function [annotations, label_sequences] = import_mediaeval_annotation(annotations_exp_path, subject_id_regexp)
    annotations = [];
    label_sequences = [];
    %annotation_files_exp = strcat(annotations_exp_path,'*.csv');
    %subject_id_regexp = '.*ann(\d+)\.csv';

    % Read in annotation data for each subject, task, session, and label type
    [annotation_file_path,annotation_file_name,annotation_file_ext] = fileparts(annotations_exp_path);
    annotations_files = dir(annotation_file_path);
    num_files = length(annotations_files);

    % Read in the raw annotations
    for file_index=1:length(annotations_files)
        file_name = annotations_files(file_index).name;

        % Ignore files not requested by user input
        do_skip_file = ~strcmp(file_name, strcat(annotation_file_name, annotation_file_ext));
        if do_skip_file
            continue;
        end

        % Get the subject ID using the input reg exp
        file_token = regexp(file_name, subject_id_regexp, 'tokens');
        file_subject_id = str2num(char(file_token{1}));
        if isempty(file_subject_id)
            disp(sprintf( 'Unable to match regular expression to file: %s. Skipping file', file_name));
            continue;
        end
        
        annotations_per_subject{file_subject_id} = struct();
        csv_data = read_csv_file(fullfile(annotation_file_path,file_name), ',');
        csv_data = cast_to_best_data_type(csv_data(2:end,2:end)); % Discard the header and worker ID column

        if ~exist('annotations', 'var') || isempty(annotations)
            annotations = csv_data';
        else
            % Make sure all annotations are the same length.
            % Truncate the end of the time series when necessary.
            if length(csv_data) > size(annotations, 2)
                annotations = [annotations; csv_data(1:size(annotations,2))'];
            elseif length(csv_data) < size(annotations, 2)
                annotations = [annotations(:,1:length(csv_data)); csv_data'];
            else
                annotations = [annotations; csv_data'];
            end
        end
    end
    
    % Create a label sequences variable (some methods need the data
    % presented in this way)
    label_sequences = cell(0);
    for label_idx=1:size(annotations,2)
        label_sequences{length(label_sequences)+1} = annotations(:,label_idx);
    end
end
