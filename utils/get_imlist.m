function [im_list, count] = get_imlist(data_dir, dataset, ftxt)
%   data_dir specifies the location of datasets e.g. Cambridge, 7Scenes
%   dataset specifies the target dataset e.g. ShopFacade, heads
%   imlist contains the absolute path of each target image
    fname = fullfile(data_dir, dataset, ftxt);
    fid = fopen (fname, 'rt');
    im_list = {};
    count = 0;
    while feof(fid) ~= 1
        line = fgetl(fid);
        count = count + 1;
        if count <= 3
            continue;
        end
        cells = strsplit(line, ' ');
        im_path = fullfile(data_dir, dataset, cells{1});
        im_list = [im_list, im_path];
    end
    count = count - 3;
    fclose(fid);
end
