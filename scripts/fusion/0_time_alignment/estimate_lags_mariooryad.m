function [mariooryad_lags] = estimate_lags_mariooryad(annotations, feature_sequences, label_sequences, max_lag_frames)
    warning('off','stats:gmdistribution:MissingData');

    addpath('/usr/local/MATLAB/custom_toolbox/PRMLT/chapter01')
    addpath(genpath([cd '/mariooryad_lag_estimation']))
    addpath(genpath([cd '/tools/SLMtools']))

    if ~isempty(feature_sequences)
        % BB - Due to sampling issues, the labels sequences are
        % sometimes a bit longer than they should be.
        % estimate_lag(...) needs the features to be at least
        % as big as the labels, so we clip excess labels here.
        % BB - Also some of the labels do not extend to the end
        % of the features time series, so we repeat the last label so all
        % labels have the same max length
        max_label_len = length(feature_sequences{1});
        for label_sequence_idx=1:length(label_sequences)
            len = length(label_sequences{label_sequence_idx});
            if len > max_label_len
                label_sequences{label_sequence_idx} = label_sequences{label_sequence_idx}(1:max_label_len);
            elseif len < max_label_len
                label_sequences{label_sequence_idx} = [label_sequences{label_sequence_idx}; repmat(label_sequences{label_sequence_idx}(end), max_label_len-len, 1)];
            end
        end

        % Duplicate feature sequence for each annotator sequence
        for feature_seq_idx=2:length(label_sequences)
            feature_sequences{feature_seq_idx} = feature_sequences{1};
        end

        % Estimate best window lag for each annotator independently
        annotator_lags = {};
        for seq_idx=1:length(label_sequences)
            [best_ann_lag, best_ann_mutual_info] = estimate_lag(feature_sequences(seq_idx), label_sequences(seq_idx), max_lag_frames);
            annotator_lags{seq_idx} = [best_ann_lag, best_ann_mutual_info];
        end

        % Estimate single best window lag using all sequences jointly
        [best_joint_window_lag, best_joint_mutual_info] = estimate_lag(feature_sequences, label_sequences, max_lag_frames);
        joint_lags = [best_joint_window_lag, best_joint_mutual_info];

        mariooryad_lags.annotator_lags = annotator_lags;
        mariooryad_lags.joint_lag = joint_lags;
    end
end
