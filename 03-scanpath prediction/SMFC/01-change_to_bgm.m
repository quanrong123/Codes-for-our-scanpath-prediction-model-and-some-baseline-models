clear all; clc;
matspath = 'F:\scanpath prediction\comparison\20200903\CAT2000\GBVS\saliency_maps\';
matspathDir = dir([matspath, '*']);
num_mats = length(matspathDir);
resultspath = 'F:\scanpath prediction\comparison\20200903\CAT2000\GBVS\saliency maps_pgm\';
mkdir(resultspath)
for k = 3:num_mats
    mat_name = matspathDir(k).name;
    imgspath = [matspath, mat_name, '/'];
    imgspathDir = dir([imgspath, '*.jpg']);
    num_imgs = length(imgspathDir);
    res_path = [resultspath, mat_name, '/'];
    mkdir(res_path);
    for i = 1:num_imgs
        image_name = imgspathDir(i).name;
        image = imread([imgspath, image_name]);
        res_name = [res_path, image_name(1:end-4), '.pgm'];
        imwrite(image, res_name);
    end

end
