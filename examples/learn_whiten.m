%---------------------------------------------------------------------
% Setting params
%---------------------------------------------------------------------
% Set data folder, change if you have downloaded the data somewhere else
data_root = fullfile(get_root_cnnimageretrieval(), 'data');

% Check, and, if necessary, download train data (for whiten) and fine-tuned networks
%download_customize(data_root);

% Set folder of images to learn whitening 
ims_whiten_dir = fullfile(data_root, 'train', 'ims');
test_imdim = 1024;  % choose test image dimensionality

% Load training data filenames and pairs for whitening
train_whiten_file = fullfile(data_root, 'train', 'dbs', 'retrieval-SfM-30k-whiten.mat'); % less images, faster
%train_whiten_file = fullfile(data_root, 'train', 'dbs', 'retrieval-SfM-120k-whiten.mat'); % more images, a bit better results but slower
train_whiten = load(train_whiten_file);
cids  = train_whiten.cids; 
qidxs = train_whiten.qidxs; % query indexes 
pidxs = train_whiten.pidxs; % positive indexes

% Network configuration
%network_file = fullfile(data_root, 'networks', 'retrieval-SfM-30k', 'retrievalSfM30k-siamac-alex.mat'); % fine-tuned CNN network (siamac-alex or siamac-vgg)
network_file = fullfile(data_root, 'networks', 'retrieval-SfM-30k', 'retrievalSfM30k-siamac-vgg.mat');
use_rvec = 1;  % use R-MAC, otherwise use MAC
use_gpu = 1;  % use GPU (GPUID = use_gpu), otherwise use CPU
to_extract = 1; % whether extract descriptor, if not load desc_file

%---------------------------------------------------------------------
% Load network model
%---------------------------------------------------------------------
% Prepare function for desc extraction
if ~use_rvec, descfun = @(x, y) cnn_vec (x, y);  else, descfun = @(x, y) cnn_vecr(x, y); end

[~, network_name, ~] = fileparts(network_file);
fprintf('>>Loading CNN image retrieval model %s...\n', network_name);

load(network_file);
net = dagnn.DagNN.loadobj(net);
if use_gpu,	gpuDevice(use_gpu); net.move('gpu'); end

%---------------------------------------------------------------------
% Extract/Load descriptors
%---------------------------------------------------------------------
if use_rvec, whiten_tp = 'rmac'; else, whiten_tp = 'mac'; end 
whiten_file = fullfile(data_root, 'whiten',  sprintf('Sfm30k-vgg-%s.mat',whiten_tp));

if to_extract
    % Extract CNN descriptors 
    fprintf('>> whitening: Extracting CNN descriptors for training images...\n');
    progressbar(0);
    for i=1:numel(cids)
        vecs_whiten{i} = descfun(imresizemaxd(imread(cid2filename(cids{i}, ims_whiten_dir)), test_imdim, 0), net);
        progressbar(i/numel(cids));
    end
    vecs_whiten = cell2mat(vecs_whiten);
    save(whiten_file, 'vecs_whiten');
else
    fprintf('>> Load pre-extracted descriptor...\n');
    mat = load(whiten_file, 'vecs_whiten');
    vecs_whiten = mat.vecs_whiten;
end

%---------------------------------------------------------------------
% Learn whitening
%---------------------------------------------------------------------
fprintf('>> whitening: Learning...\n');
[Lw, PCAw] = whitenlearn(vecs_whiten, qidxs, pidxs);
save(whiten_file, 'Lw', '-append');
save(whiten_file, 'PCAw', '-append');

