function mutual_info = estimate_mutual_information(feature, label)
%Estimates mutual information between a set of features and a set of
%continuous labels, assuming normal distribution.  
%   feature: a sample_count x feature_count matrix
%   label: a sample_count x label_count matrix
  sample_count = size(feature, 1);
  mixture_count = 1;
  gmm_label_feature = gmdistribution.fit([label feature], mixture_count);
  gmm_feature = gmdistribution.fit(feature, mixture_count);
  gmm_label = gmdistribution.fit(label, mixture_count);
  
  % BB - Using NlogL for the determinant of the covariance matrix doesn't
  % make sense.
  % EDIT - It's not a great mutual information approximation, but it still
  % works.
  mutual_info = (-gmm_label_feature.NlogL + gmm_label.NlogL + gmm_feature.NlogL) / sample_count;
  %mutual_info = -0.5*log(det(gmm_feature.Sigma)*det(gmm_label.Sigma)/det(gmm_label_feature.Sigma));
end
