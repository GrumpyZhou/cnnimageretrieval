clear;

%---------------------------------------------------------------------
% Set data folder and testing parameters
%---------------------------------------------------------------------

% Set data folder, change if you have downloaded the data somewhere else
data_root = fullfile(get_root_cnnimageretrieval(), 'data');

% Set test options
test_datasets = {'paris6k'};  % list of datasets to evaluate on
test_imdim = 1024;  % choose test image dimensionality

% Network configuration
%network_file = fullfile(data_root, 'networks', 'retrieval-SfM-30k', 'retrievalSfM30k-siamac-alex.mat'); % fine-tuned CNN network (siamac-alex or siamac-vgg)
network_file = fullfile(data_root, 'networks', 'retrieval-SfM-30k', 'retrievalSfM30k-siamac-vgg.mat');
use_rvec = 0;  % use R-MAC, otherwise use MAC
use_gpu = 1;  % use GPU (GPUID = use_gpu), otherwise use CPU

%---------------------------------------------------------------------
% Load network model
%---------------------------------------------------------------------
% Prepare function for desc extraction
if ~use_rvec, descfun = @(x, y) cnn_vec (x, y);  else, descfun = @(x, y) cnn_vecr(x, y); end

[~, network_name, ~] = fileparts(network_file);
fprintf('>>Loading CNN image retrieval model %s...\n', network_name);

load(network_file);
net = dagnn.DagNN.loadobj(net);
if use_gpu,     gpuDevice(use_gpu); net.move('gpu'); end

%---------------------------------------------------------------------
% Load whitening variables
%---------------------------------------------------------------------
% Choose training data for whitening and set up data folder
if use_rvec, whiten_tp = 'rmac'; else, whiten_tp = 'mac'; end
whiten_file = fullfile(data_root, 'whiten',  sprintf('Sfm120k-vgg-%s.mat',whiten_tp));
whiten = load(whiten_file);
Lw = whiten.Lw;

%---------------------------------------------------------------------
% Extract descriptor for testing imgs and evaluate
%---------------------------------------------------------------------

% extract and evaluate
for d = 1:numel(test_datasets)
	fprintf('>> %s: Processing test dataset...\n', test_datasets{d});		
	cfg = configdataset (test_datasets{d}, fullfile(data_root, 'test/')); % config file for the dataset

	fprintf('>> %s: Extracting CNN descriptors for db images...\n', test_datasets{d}); 
	progressbar(0); vecs = [];
	for i = 1:cfg.n
		vecs{i} = descfun(imresizemaxd(imread(cfg.im_fname(cfg, i)), test_imdim, 0), net);
		progressbar(i/cfg.n);
	end
	vecs = cell2mat(vecs);

	fprintf('>> %s: Extracting CNN descriptors for query images...\n', test_datasets{d}); 
	progressbar(0); qvecs = [];
	for i = 1:cfg.nq
		qvecs{i} = descfun(crop_qim(imread(cfg.qim_fname(cfg, i)), cfg.gnd(i).bbx, test_imdim), net);
		progressbar(i/cfg.nq);
	end
	qvecs = cell2mat(qvecs);

	vecsLw = whitenapply(vecs, Lw.m, Lw.P); % apply whitening on database descriptors
	qvecsLw = whitenapply(qvecs, Lw.m, Lw.P); % apply whitening on query descriptors

	fprintf('>> %s: Retrieval...\n', test_datasets{d});
	% raw descriptors
	sim = vecs'*qvecs;
	[sim, ranks] = sort(sim, 'descend');
	map = compute_map (ranks, cfg.gnd);	
	fprintf('>> %s: mAP = %.4f, without whiten\n', test_datasets{d}, map);
	% with learned whitening
	sim = vecsLw'*qvecsLw;
	[sim, ranks] = sort(sim, 'descend');
	map = compute_map (ranks, cfg.gnd);	
	fprintf('>> %s: mAP = %.4f, with whiten\n', test_datasets{d}, map);
end
