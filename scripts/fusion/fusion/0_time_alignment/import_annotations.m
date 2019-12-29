function [annotations, label_sequences] = import_annotations(task_name, frequency)
    annotations = [];
    label_sequences = [];
    annotation_files_exp = strcat('../../../datasets/green_intensity/',task_name,'/annotations_',num2str(frequency),'hz/*.csv');
    subject_id_regexp = '.*ann(\d+)\.csv';

    % Read in annotation data for each subject, task, session, and label type
    [annotation_file_path,name,ext] = fileparts(annotation_files_exp);
    annotations_files = dir(annotation_files_exp);
    num_subjects = length(annotations_files);

    % Read in the raw annotations
    for subject_id=1:num_subjects
        annotations_per_subject{subject_id} = struct();
        for file_index=1:length(annotations_files)
            file_name = annotations_files(file_index).name;
            file_token = regexp(file_name, subject_id_regexp, 'tokens');
            file_subject_id = str2num(char(file_token{1}));
            if isempty(file_subject_id)
                disp(sprintf( 'Unable to match regular expression to file: %s. Skipping file', file_name));
                continue;
            end
            if file_subject_id == subject_id
                csv_data = read_csv_file(fullfile(annotation_file_path,file_name), ',');
                csv_data = cast_to_best_data_type(csv_data(2:end,2:end)); % Discard the header and time column

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
        end
    end

    % Create a label sequences variable (some methods need the data
    % presented in this way)
    label_sequences = cell(0);
    for subject_id=1:num_subjects
        label_sequences{length(label_sequences)+1} = annotations(subject_id,:)';
    end
end
