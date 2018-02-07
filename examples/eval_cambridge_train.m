clear;

%---------------------------------------------------------------------
% Set data folder and testing parameters
%---------------------------------------------------------------------

% Set data folder, change if you have downloaded the data somewhere else
data_root = fullfile(get_root_cnnimageretrieval(), 'data');
cambridge_root = '/usr/stud/zhouq/CambridgeLandmark';

% Set test options
test_datasets = {'ShopFacade', 'KingsCollege', 'OldHospital', 'StMarysChurch'};  % list of datasets to evaluate on
%---------------------------------------------------------------------
% Load whitening variables
%---------------------------------------------------------------------
% Choose training data for whitening and set up data folder
whiten_file = fullfile(data_root, 'whiten', 'Sfm120k-vgg-mac.mat');
whiten = load(whiten_file);
Lw = whiten.Lw;

%---------------------------------------------------------------------
% Extract descriptor for testing imgs and evaluate
%---------------------------------------------------------------------
% Set path to store the result
result_dir = fullfile(data_root, 'cambridge-train', 'vgg-mac-1024');
if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end
% extract and evaluate
for d = 1:numel(test_datasets)
    dataset = test_datasets{d};
    desc_file = fullfile(result_dir,  sprintf('%s.mat', dataset));
    fprintf('>> %s: Load pre-saved descriptors for training and query/test images...\n', dataset); 
    res = load(desc_file);
    vecs = res.vecs;
    qvecs = res.vecs; % queries images are also training images

    vecsLw = whitenapply(vecs, Lw.m, Lw.P); % apply whitening on database descriptors
    qvecsLw = whitenapply(qvecs, Lw.m, Lw.P); % apply whitening on query descriptors

    % Find k-nearest neighbours
    fprintf('>> %s: K-NearestNeighbour Retrieval...\n', dataset);
    % with learned whitening
    sim = vecsLw'*qvecsLw;
    [sim, ranks] = sort(sim, 'descend');
    idx_file = fullfile(result_dir,  sprintf('%s-knn.mat', dataset));
    save(idx_file, 'sim');
    save(idx_file, 'ranks', '-append');
    fprintf('>> Save knn result to %s', idx_file);
end

