truth_signal_csv = '/USC/2016_Continuous_Annotations/annotation_tasks/TaskA/AnnotationData/objective_truth/TaskA_normalized_1hz.csv';
obj_truth = csvread(truth_signal_csv, 1, 1);

%% Generate triplets such that for each (i,j,k), the obj_truth at index i is closer to k than j
n = length(obj_truth);
num_triplets = n*(n-1)*(n-2)/2;
num_triplets = 2*num_triplets; % Worst case if all comparisons are equal
triplets = zeros(num_triplets,3);
triplet_idx = 1;
diff_eps = 0.01;
for i=1:n
    for j=1:n
        if i == j
            continue;
        end
        for k=j+1:n
            if i == k
                continue;
            end
            diff_ij = norm(obj_truth(i)-obj_truth(j));
            diff_ik = norm(obj_truth(i)-obj_truth(k));
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

triplets = triplets(1:triplet_idx-1,:);

embedding = computeEmbedding(triplets, 1.0, 1.0);

time = 1:length(obj_truth);
time = time / 1.0; % 1 Hz
plot(time, obj_truth, 'r-'); hold on;
plot(time, embedding, 'b--');