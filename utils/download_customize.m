function download_customize(data_dir)
% DOWNLOAD_TRAIN Checks, and, if required, downloads the necessary data and networks for the training.
%   Folder structure:
%     DATA_ROOT/train/dbs/         : folder with training database mat files
%     DATA_ROOT/train/ims/         : folder with original images used for training
%     DATA_ROOT/networks/retrieval-SfM-30k/ : CNN models fine-tuned for image retrieval using retrieval-SfM-30k data

    % Create data folder if it does not exist
    if ~exist(data_dir, 'dir')
        mkdir(data_dir);
    end

    % Create train folder if it does not exist
    train_dir = fullfile(data_dir, 'train');
    if ~exist(train_dir, 'dir')
        mkdir(train_dir);
    end
       
   % Download folder original images in train/ims/
    src_dir = fullfile('http://cmp.felk.cvut.cz/cnnimageretrieval/data', 'train', 'ims');
    dst_dir = fullfile(data_dir, 'train', 'ims');
    dl_file = 'ims.tar.gz';
    if ~exist(dst_dir, 'dir')
        src_file = fullfile(src_dir, dl_file);
        dst_file = fullfile(dst_dir, dl_file);
        fprintf('>> Image directory does not exist. Creating: %s\n', dst_dir);
        mkdir(dst_dir);
        fprintf('>> Downloading ims.tar.gz...\n');
        system(sprintf('wget %s -O %s', src_file, dst_file));
        fprintf('>> Extracting %s...\n', dst_file);
        system(sprintf('tar -zxf %s -C %s', dst_file, dst_dir));
        fprintf('>> Extracted, deleting %s...\n', dst_file);
        system(sprintf('rm %s', dst_file));
    end
 
    % Download retrieval SfM whitening mat file in train/db/
    src_dir = fullfile('http://cmp.felk.cvut.cz/cnnimageretrieval/data', 'train', 'dbs');
    dst_dir = fullfile(data_dir, 'train', 'dbs');
    dl_files = {'retrieval-SfM-30k-whiten.mat', ...
                'retrieval-SfM-120k-whiten.mat'};
    if ~exist(dst_dir, 'dir')
        fprintf('>> Database directory does not exist. Creating: %s\n', dst_dir);
        mkdir(dst_dir);
        fprintf('>> Downloading database files from cmp.felk.cvut.cz/cnnimageretrieval\n');
    end
    for i = 1:numel(dl_files)
        src_file = fullfile(src_dir, dl_files{i});
        dst_file = fullfile(dst_dir, dl_files{i});
        if ~exist(dst_file, 'file')
            fprintf('>> DB file %s does not exist. Downloading...\n', dl_files{i});
            system(sprintf('wget %s -O %s', src_file, dst_file)); 
        end
    end
  
    % Download fine-tuned models in networks/retrieval-SfM-30k/
    src_dir = fullfile('http://cmp.felk.cvut.cz/cnnimageretrieval/data', 'networks', 'retrieval-SfM-30k');
    dst_dir = fullfile(data_dir, 'networks', 'retrieval-SfM-30k');
    dl_files = {'retrievalSfM30k-siamac-alex.mat', 'retrievalSfM30k-siamac-vgg.mat'};
    if ~exist(dst_dir, 'dir')
        fprintf('>> Fine-tuned networks directory does not exist. Creating: %s\n', dst_dir);
        mkdir(dst_dir);
        fprintf('>> Downloading fine-tuned network files from http://cmp.felk.cvut.cz/cnnimageretrieval\n');
    end
    for i = 1:numel(dl_files)
        src_file = fullfile(src_dir, dl_files{i});
        dst_file = fullfile(dst_dir, dl_files{i});
        if ~exist(dst_file, 'file')
            fprintf('>> Network %s does not exist. Downloading...\n', dl_files{i});
            system(sprintf('wget %s -O %s', src_file, dst_file)); 
        end
    end

