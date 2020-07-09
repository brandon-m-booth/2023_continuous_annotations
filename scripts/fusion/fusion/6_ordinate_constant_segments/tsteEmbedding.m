function tsteEmbedding(triplets_file_path, output_file_path)
% Generates an embedding using t-STE and simulated triplet comparisons.  The
% triplets are created from the approximate signal over the constant intervals
% by peaking at the true signal.  The correctness_rate affects the probability
% of not adversarially flipping each triplet comparison and the
% comparison_retain_percentage is the fraction of all possible triplets
% that are used to construct the embedding.
    [file_path,file_name,ext] = fileparts(mfilename('fullpath'));
    addpath(strcat(file_path,'/tste'));

    if ~exist(triplets_file_path, 'file')
      return
    end

    %% Load data
    triplets = csvread(triplets_file_path);
    if min(min(triplets)) == 0
        triplets = triplets + 1;
    end

    embedding = computeEmbedding(triplets, 1.0, 1.0);

    %% Rescale each embedding uniformly to [0,1] interval
    embedding = embedding-mean(embedding);
    max_emb = max(abs(embedding));
    embedding = 0.5*embedding/max_emb + 0.5;

    csvwrite(output_file_path, embedding);
