clear;
close all;
clc;

% params for generating synthetic data with a synthetic shift between
% features and labels.
sequence_count = 100;
min_feature_count = 5;
max_feature_count = 50;
label_count = 1;
min_sequence_length = 200;
max_sequence_length = 500;
max_lag = 100;
ground_truth_lag = 25;

feature_sequences = cell(sequence_count, 1);
label_sequences = cell(sequence_count, 1);
feature_count = min_feature_count - 1 ...
        + randi(max_feature_count - min_feature_count + 1);
feat_label_map = randn(feature_count, label_count);

for i=1:sequence_count
    sequence_len = min_sequence_length - 1 ...
        + randi(max_sequence_length - min_sequence_length + 1);
    feature_sequences{i} = cumsum(rand(sequence_len, feature_count));
    label_sequences{i} = feature_sequences{i} * feat_label_map ...
        + randn(sequence_len, label_count);
    label_sequences{i} = label_sequences{i}(...
        [end-ground_truth_lag+1:end 1:end-ground_truth_lag],:);
end
lag = estimate_lag(feature_sequences, label_sequences, max_lag);
disp(['Synthetic Lag = ' mat2str(ground_truth_lag) ' frames, Estimated Lag = ' mat2str(lag) ' frames.']);
