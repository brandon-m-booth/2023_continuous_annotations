function [lag, max_val] = estimate_lag(feature_sequences, label_sequences, max_lag)
%Estimates lag in labels compared to features based on mutual information.  
%   feature_sequences: a cell array of feature sequence
%   label_sequences: a cell array of continuous label corresponding the
%   features.
%   max_lag: maximum possible lag in number of frames

do_plot = 0;

sequence_count = length(feature_sequences);
feature_count = size(feature_sequences{1}, 2);
label_count = size(label_sequences{1}, 2);

max_len = 0;
for sequence_index=1:sequence_count
    max_len = max_len + size(feature_sequences{sequence_index}, 1);
end
mutual_informations = zeros(max_lag + 1, 1);
shifts = 0:max_lag;
for shift_index=1:length(shifts)
    shift = shifts(shift_index);
    features = zeros(max_len, feature_count);
    labels = zeros(max_len, label_count);
    index = 1;
    for sequence_index=1:sequence_count
        label = label_sequences{sequence_index}(1+shift:end,:);
        len = size(label, 1);
        feature = feature_sequences{sequence_index}(1:len,:);
        features(index:index+len-1, :) = feature;
        labels(index:index+len-1, :) = label;
        index = index + len;
    end
    features(index:end,:) = [];
    labels(index:end,:) = [];
    try
        mutual_informations(shift_index) = estimate_mutual_information(features, labels);
    catch
        mutual_informations(shift_index) = 0;
    end
    %disp(['Lag = ' mat2str(shifts(shift_index)) ' frames => Mutual Info = ' mat2str(mutual_informations(shift_index))]);
end
[max_val, max_index] = max(mutual_informations);
if do_plot
    plot(shifts, mutual_informations);
    hold on;
    plot([shifts(max_index) shifts(max_index)], [0 max_val],'--');
    xlabel('frame shift');
    ylabel('mutual information');
    text(shifts(max_index), max_val, ['Estimated Lag = ' mat2str(shifts(max_index)) ' frames']);
end
lag = shifts(max_index);
end
