clear all; clc; 
labelspath = '/disk4/quanrong/rebuttal_2021/MIT/BayesDGC_result/';
spspath = '/disk4/quanrong/rebuttal_2021/MIT/MIT_sp_seg_2000_1/';
resultpath = '/disk4/quanrong/rebuttal_2021/MIT/BayesDGC_result_final/';
mkdir(resultpath);
imgspath = '/disk4/quanrong/rebuttal_2021/MIT/images/';
imgspathdir = dir([imgspath, '*.jpg']);
labelspathdir = dir([labelspath, '*.mat']);
spspathdir = dir([labelspath, '*.mat']);
num_images = length(labelspathdir);
for i=1:num_images
    image_name = labelspathdir(i).name;
    res_path1 = [resultpath, image_name(1:end-4), '_b.jpg'];
    res_path2 = [resultpath, image_name(1:end-4), '_s.jpg'];
    label_path = [labelspath, image_name];
    sp_path = [spspath, image_name];
    load(label_path);
    load(sp_path);
    image_b = zeros(600, 800);
    image_s = zeros(600, 800);
    num_sp = length(spstats);
    image = imread([imgspath, imgspathdir(i).name]);
    [h w c] = size(image);
    for j = 1:num_sp
        image_b(find(sp_seg==j)) = prediction_b(j); 
        image_s(find(sp_seg==j)) = prediction_s(j);
    end
    %imwrite(image_b, res_path1);
    image_s1 = imresize(image_s, [h w]);
    imwrite(image_s1, res_path2);
end












