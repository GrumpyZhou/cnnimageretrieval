clear;

%---------------------------------------------------------------------
% Set data folder and testing parameters
%---------------------------------------------------------------------

% Set data folder, change if you have downloaded the data somewhere else
data_root = fullfile(get_root_cnnimageretrieval(), 'data');
cambridge_root = '/usr/stud/zhouq/CambridgeLandmark';

% Set test options
test_datasets = {'ShopFacade', 'KingsCollege', 'OldHospital', 'StMarysChurch'};  % list of datasets to evaluate on
test_imdim = 1024;  % Resizes image so that longer edge is maximum to the given size

% Network configuration
network_file = fullfile(data_root, 'networks', 'retrieval-SfM-30k', 'retrievalSfM30k-siamac-vgg.mat');
use_rvec = 0;  % Use R-MAC, otherwise use MAC
use_gpu = 1;  % Use GPU (GPUID = use_gpu), otherwise use CPU

%---------------------------------------------------------------------
% Load network model
%---------------------------------------------------------------------
if ~use_rvec, descfun = @(x, y) cnn_vec (x, y);  else, descfun = @(x, y) cnn_vecr(x, y); end
[~, network_name, ~] = fileparts(network_file);
fprintf('>>Loading CNN image retrieval model %s...\n', network_name);
load(network_file);
net = dagnn.DagNN.loadobj(net);
if use_gpu,     gpuDevice(use_gpu); net.move('gpu'); end

%---------------------------------------------------------------------
% Extract descriptor for both training and testing imgs
%---------------------------------------------------------------------
if use_rvec, desc_tp = 'rmac'; else, desc_tp = 'mac'; end
result_dir = fullfile(data_root, 'cambridge', sprintf('vgg-%s-%d-desc', desc_tp, test_imdim));
if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end
for d = 1:numel(test_datasets)
    dataset = test_datasets{d};
    fprintf('>> %s: Processing test dataset...\n', dataset);
    desc_file = fullfile(result_dir,  sprintf('%s.mat', dataset));    

    % Extract training images descriptors
    [train_im, n] = get_cambridge_imlist(cambridge_root, dataset, 'dataset_train.txt'); 
    fprintf('>> %s: Extracting CNN descriptors for training images...\n', dataset); 
    progressbar(0); vecs = [];   
    for i = 1:n
        vecs{i} = descfun(imresizemaxd(imread(train_im{i}), test_imdim, 0), net);
        progressbar(i/n);
    end
    vecs = cell2mat(vecs);
    save(desc_file, 'vecs');

    % Extract training images descriptors
    [test_im, n] = get_cambridge_imlist(cambridge_root, dataset, 'dataset_test.txt');
    fprintf('>> %s: Extracting CNN descriptors for query/test images...\n', dataset); 
    progressbar(0); qvecs = [];	
    for i = 1:n
        qvecs{i} = descfun(imresizemaxd(imread(test_im{i}), test_imdim, 0), net);
        progressbar(i/n);
    end
    qvecs = cell2mat(qvecs);
    save(desc_file, 'qvecs', '-append');   

    % Save feature vectors together in one .mat file
    fprintf('>> Save cnn descriptors to %s', desc_file);
end
