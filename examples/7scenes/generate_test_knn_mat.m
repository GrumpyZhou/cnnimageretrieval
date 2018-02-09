clear;

%---------------------------------------------------------------------
% Setup datasets
%---------------------------------------------------------------------
data_root = fullfile(get_root_cnnimageretrieval(), 'data');
test_datasets = {'heads', 'chess', 'fire', 'office', 'pumpkin', 'redkitchen', 'stairs'};  % list of datasets to evaluate on
%---------------------------------------------------------------------
% Load whitening variables
%---------------------------------------------------------------------
whiten_file = fullfile(data_root, 'whiten', 'Sfm120k-vgg-mac.mat');
whiten = load(whiten_file);
Lw = whiten.Lw;

%---------------------------------------------------------------------
% Calculate similarity matrix for training pairs
%---------------------------------------------------------------------
% Set path to store the result
result_dir = fullfile(data_root, '7scenes', 'test');
if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end
for d = 1:numel(test_datasets)
    dataset = test_datasets{d};
    desc_file = fullfile(result_dir, 'vgg-mac-1024-desc', sprintf('%s.mat', dataset));
    fprintf('>> %s: Load pre-saved descriptors for training and query/test images...\n', dataset); 
    res = load(desc_file);
    vecs = res.vecs;
    qvecs = res.qvecs; 
    vecsLw = whitenapply(vecs, Lw.m, Lw.P); % Apply whitening on database descriptors
    qvecsLw = whitenapply(qvecs, Lw.m, Lw.P); % Apply whitening on query descriptors

    % Find k-nearest neighbours by calculating similarity 
    fprintf('>> %s: K-NearestNeighbour Retrieval...\n', dataset);
    sim = vecsLw'*qvecsLw;
    [sim, ranks] = sort(sim, 'descend');
    idx_file = fullfile(result_dir, sprintf('%s-knn.mat', dataset));
    save(idx_file, 'sim');
    save(idx_file, 'ranks', '-append');
    fprintf('>> Save knn result to %s', idx_file);
end
