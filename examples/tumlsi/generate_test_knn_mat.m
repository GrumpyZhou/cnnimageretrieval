clear;

%---------------------------------------------------------------------
% Setup datasets
%---------------------------------------------------------------------
data_root = fullfile(get_root_cnnimageretrieval(), 'data');
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
base_dir = fullfile(data_root, 'tumlsi');
result_dir = fullfile(base_dir, 'test');
if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

desc_file = fullfile(base_dir, 'vgg-mac-1024-desc', 'tumlsi.mat');
fprintf('>> Load pre-saved descriptors for query/test images...\n');
res = load(desc_file);
vecs = res.vecs;
qvecs = res.qvecs; 
vecsLw = whitenapply(vecs, Lw.m, Lw.P); % Apply whitening on database descriptors
qvecsLw = whitenapply(qvecs, Lw.m, Lw.P); % Apply whitening on query descriptors

% Find k-nearest neighbours by calculating similarity 
fprintf('>> K-NearestNeighbour Retrieval...\n');
sim = vecsLw'*qvecsLw;
[sim, ranks] = sort(sim, 'descend');
idx_file = fullfile(result_dir, 'tumlsi-knn.mat');
save(idx_file, 'sim');
save(idx_file, 'ranks', '-append');
fprintf('>> Save knn result to %s', idx_file);

