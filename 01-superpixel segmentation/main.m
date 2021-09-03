clear all; clc;
imgspath = 'H:\SLIC-segmentation\MIT_images_bmp\';
imgspathdir = dir([imgspath, '*.bmp']);
num_imgs = length(imgspathdir);
seg_map_path = 'H:\SLIC-segmentation\MIT_bmp_seg_2000_1\';
mkdir(seg_map_path);
save_path = 'H:\SLIC-segmentation\MIT_sp_seg_2000_1\';
mkdir(save_path);
for i = 1:num_imgs
    img_path = [imgspath, imgspathdir(i).name];
    system(['SLICSuperpixelSegmentation ',img_path,' 20 2000 ',seg_map_path]);
    im=imread(img_path);
    dat_path = [seg_map_path, imgspathdir(i).name(1:end-4), '.dat'];
    mat_path = [save_path, imgspathdir(i).name(1:end-4), '.mat'];
    sp_seg = ReadDAT(size(im), dat_path);
    spstats = regionprops(sp_seg, 'PixelIdxList');
    save(mat_path, 'sp_seg','spstats')
end






















