function oracleTsteEmbedding(output_file_path, truth_signal_csv, intervals_csv, comparison_retain_percentage, correctness_rate)
% Generates an embedding using t-STE and simulated triplet comparisons.  The
% triplets are created from the approximate signal over the constant intervals
% by peaking at the true signal.  The correctness_rate affects the probability
% of not adversarially flipping each triplet comparison and the
% comparison_retain_percentage is the fraction of all possible triplets
% that are used to construct the embedding.
    [file_path,file_name,ext] = fileparts(mfilename('fullpath'));
    addpath(strcat(file_path,'/tste'));
    if nargin <= 4
        comparison_retain_percentage = 1.0;
        correctness_rate = 1.0;
    elseif nargin <= 5
        correctness_rate = 1.0;
    end

    addpath(strcat(mfilename('fullpath'), 'tste'));

    if ~exist(truth_signal_csv, 'file') || ~exist(intervals_csv, 'file')
      return
    end

    %% Load data
    obj_truth = csvread(truth_signal_csv, 1, 1);
    intervals = csvread(intervals_csv);

    %% For each interval, compute the average obj truth value
    obj_mean = zeros(size(intervals,1),1);
    for i=1:size(intervals,1)
        interval = intervals(i,:);
        obj_mean(i) = mean(obj_truth(interval(1)+1:interval(2)+1));
    end

    %% Generate triplets such that for each (i,j,k), the obj_truth at index i is closer to k than j
    n = size(intervals,1);
    num_triplets = n*(n-1)*(n-2)/2;
    num_triplets = 2*num_triplets; % Worst case if all comparisons are equal
    triplets = zeros(num_triplets,3);
    triplet_idx = 1;
    diff_eps = 0.01;
    for i=1:size(intervals,1)
        for j=1:size(intervals,1)
            if i == j
                continue;
            end
            for k=j+1:size(intervals,1)
                if i == k
                    continue;
                end
                diff_ij = norm(obj_mean(i)-obj_mean(j));
                diff_ik = norm(obj_mean(i)-obj_mean(k));
                if abs(diff_ik - diff_ij) < diff_eps
                    % If similar, add one triplet for both cases
                    %triplets(triplet_idx,:) = [i,k,j];
                    %triplet_idx = triplet_idx + 1;
                    %triplets(triplet_idx,:) = [i,j,k];
                    %triplet_idx = triplet_idx + 1;
                    %if i == 20
                    %   triplets(triplet_idx,:) = [i,j,k];
                    %   triplet_idx = triplet_idx + 1;
                    %end
                elseif diff_ik < diff_ij
                    triplets(triplet_idx,:) = [i,k,j];
                    triplet_idx = triplet_idx + 1;
                else
                    triplets(triplet_idx,:) = [i,j,k];
                    triplet_idx = triplet_idx + 1;
                end
            end
        end
    end

    %% Remove rows with all zeros
    [i,j] = find(triplets);
    triplets = triplets(unique(i),:);

    embedding = computeEmbedding(triplets, comparison_retain_percentage, correctness_rate)

    %% Rescale each embedding uniformly to [0,1] interval and flip if necessary
    mean_emb = mean(embedding);
    max_emb = max(abs(embedding-mean_emb));
    embedding = (embedding-mean_emb)/max_emb + mean_emb;
    if corr(embedding,obj_mean-mean(obj_mean)) >= 0
        embedding = 0.5*embedding + 0.5;
    else
        embedding = -0.5*embedding + 0.5;
    end

    csvwrite(output_file_path, embedding);

