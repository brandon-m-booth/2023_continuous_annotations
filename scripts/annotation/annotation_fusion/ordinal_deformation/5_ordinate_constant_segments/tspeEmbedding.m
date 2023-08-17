function tspeEmbedding(pairs_file_path, output_file_path)
    [file_path,file_name,ext] = fileparts(mfilename('fullpath'));
    addpath(strcat(file_path,'/tste'));

    if ~exist(pairs_file_path, 'file')
      return
    end

    %% Load data
    pairs = csvread(pairs_file_path);
    if min(min(pairs)) == 0
        pairs = pairs + 1;
    end
    
    %% Make a dummy sample to act as the reference in triplet embedding
    next_sample_idx = max(max(pairs))+1;
    triplets = [repmat([next_sample_idx], size(pairs,1), 1), pairs];
    %dummy_pairs = [repmat([next_sample_idx], next_sample_idx-1, 1),[1:next_sample_idx-1]'];
    %pairs = [pairs; dummy_pairs];

    embedding = computeEmbedding(triplets, 1.0, 1.0);
    
    if embedding(end) > mean(embedding)
        embedding = embedding(1:end-1);
        embedding = -embedding;
    else
        embedding = embedding(1:end-1);
    end

    %% Rescale each embedding uniformly to [0,1] interval
    embedding = embedding-mean(embedding);
    max_emb = max(abs(embedding));
    embedding = 0.5*embedding/max_emb + 0.5;

    csvwrite(output_file_path, embedding);
