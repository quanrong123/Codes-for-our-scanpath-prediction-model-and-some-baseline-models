clear all; clc; 
threshold = 2;
imgspath = '/disk4/quanrong/rebuttal_2021/MIT/images_1/';
sal1path = '/disk4/quanrong/rebuttal_2021/MIT/saliency_maps_new/DVA/';
sal2path = '/disk4/quanrong/rebuttal_2021/MIT/saliency_maps_new/iSEEL/';
sal3path = '/disk4/quanrong/rebuttal_2021/MIT/saliency_maps_new/MLNet/';
sal4path = '/disk4/quanrong/rebuttal_2021/MIT/saliency_maps_new/SalGAN/';
sp_seg_path = '/disk4/quanrong/rebuttal_2021/MIT/MIT_sp_seg_2000_1/';
imgspathdir = dir([imgspath, '*.jpg']);
num_imgs = length(imgspathdir);
save_path = '/disk4/quanrong/rebuttal_2021/MIT/label_data_2000_10_1/';
mkdir(save_path);
for i = 1:num_imgs
    sal1 = im2double(imread([sal1path, imgspathdir(i).name]));
    sal2 = im2double(imread([sal2path, imgspathdir(i).name]));
    sal3 = im2double(imread([sal3path, imgspathdir(i).name]));
    sal4 = im2double(imread([sal4path, imgspathdir(i).name]));
    
    sal1 = imresize(sal1, [600,800]);
    sal2 = imresize(sal2, [600,800]);
    sal3 = imresize(sal3, [600,800]);
    sal4 = imresize(sal4, [600,800]);
    
    l1 = get_labels(sal1);
    l2 = get_labels(sal2); 
    l3 = get_labels(sal3);
    l4 = get_labels(sal4);
    
    load([sp_seg_path, imgspathdir(i).name(1:end-4), '.mat']);
    spnum = length(spstats);
    L = zeros(spnum, 4);
    for j = 1:spnum
        sp_index = spstats(j).PixelIdxList;
        L(j,1) = mode(l1(sp_index));
        L(j,2) = mode(l2(sp_index));
        L(j,3) = mode(l3(sp_index));
        L(j,4) = mode(l4(sp_index));
    end   
     
    save_name = [save_path, imgspathdir(i).name(1:end-4), '.mat'];
    save(save_name, 'true_labels');
end

































