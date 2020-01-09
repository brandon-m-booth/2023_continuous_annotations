function [embedding] = computeEmbedding(triplets, comparison_retain_percentage, correctness_rate, initial_condition)
% Generates an embedding using t-STE and the input triplet matrix of the form:
% [i,j,k] if ||x_i-x_j|| < ||x_i-x_k||.  The correctness_rate affects the probability
% of not adversarially flipping each triplet comparison and the comparison_retain_percentage
% is the fraction of all possible triplets that are used to construct the embedding.

    %% Uniformly retain some percentage of the triplets
    triplets = triplets(randperm(size(triplets,1)),:);
    num_retain_triplets = round(size(triplets,1)*comparison_retain_percentage);
    triplets = triplets(1:num_retain_triplets,:);

    %% Flip the polarity of the triplet comparison for some of the triplets
    num_flips = round((1.0-correctness_rate)*size(triplets,1));
    for flip_idx=1:num_flips
        triplets(flip_idx,:) = [triplets(flip_idx,1), triplets(flip_idx,3), triplets(flip_idx,2)];
    end

    %% t-STE embedding
    d = 1;
    lambda = 0.0;
    alpha = 1.0;
    use_log = true;
    if ~exist('initial_condition', 'var')
        embedding = tste(triplets, d, lambda, alpha, use_log);    
    else
        embedding = tste(triplets, d, lambda, alpha, use_log, initial_condition);
    end