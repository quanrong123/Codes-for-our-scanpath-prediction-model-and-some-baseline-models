clear all; clc;

addpath(genpath('./'));
paras.t = 1.5;
paras.theta =200;
paras.alpha = 0.99;
paras.gamma =1;
num_fa = 4;

imgspath = '/media/quanrong/TOSHIBA/GPU2/rebuttal_2021/OSIE/images/';
sal1path = '/media/quanrong/TOSHIBA/GPU2/rebuttal_2021/OSIE/saliency_maps_new/DVA/';
sal2path = '/media/quanrong/TOSHIBA/GPU2/rebuttal_2021/OSIE/saliency_maps_new/iSEEL/';
sal3path = '/media/quanrong/TOSHIBA/GPU2/rebuttal_2021/OSIE/saliency_maps_new/MLNet/';
sal4path = '/media/quanrong/TOSHIBA/GPU2/rebuttal_2021/OSIE/saliency_maps_new/SalGAN/';

sp_seg_path = '/media/quanrong/TOSHIBA/GPU2/rebuttal_2021/OSIE/OSIE_sp_seg_2000/';
imgspathdir = dir([imgspath, '*.jpg']);
num_imgs = length(imgspathdir);
result_path = '/media/quanrong/TOSHIBA/GPU2/rebuttal_2021/OSIE/combination_GLAD_2000_3/';
mkdir(result_path);
res_path = [result_path 'res/']; 

mkdir(res_path);

threshold = 3;
for ii = 1:num_imgs
    ii
    image = im2double(imread([imgspath, imgspathdir(ii).name]));
    sal1 = im2double(imread([sal1path, imgspathdir(ii).name]));
    sal2 = im2double(imread([sal2path, imgspathdir(ii).name]));
    sal3 = im2double(imread([sal3path, imgspathdir(ii).name]));
    sal4 = im2double(imread([sal4path, imgspathdir(ii).name]));
    [height width ch] = size(image);
    
    bm1 = binary_seg(sal1, threshold);
    bm2 = binary_seg(sal2, threshold);
    bm3 = binary_seg(sal3, threshold);
    bm4 = binary_seg(sal4, threshold);

    load([sp_seg_path, imgspathdir(ii).name(1:end-4), '.mat']);
    spnum = length(spstats);
    
    [adjMatrix bd]= GetAdjMatrix(sp_seg, spnum);
    L = zeros(spnum, 4);
    for j = 1:spnum
        sp_index = spstats(j).PixelIdxList;
        L(j,1) = (sum(bm1(sp_index))/length(sp_index)) >= 0.5;
        L(j,2) = (sum(bm2(sp_index))/length(sp_index)) >= 0.5;
        L(j,3) = (sum(bm3(sp_index))/length(sp_index)) >= 0.5;
        L(j,4) = (sum(bm4(sp_index))/length(sp_index)) >= 0.5;
    end
    [sp_feat, num_fea] = Get_features(image, spstats);    
        
    %% parameters prepare
    Alpha = 0.5;
    Beta = 0.5;
    n = 0;
    %% P_ZZ = ones(spnum,1)*0.5;
    P_ZZ = zeros(spnum,length(num_fea));
    for i = 1:length(num_fea)
        num_single_fea = num_fea(i);
        fea = sp_feat(:,n+1:n+num_single_fea);
        P_Z1 = initPG2_qr( fea,L,spnum,num_fa,2);
        A = P_Z1( abs(P_Z1) ~= 9999);
        a1 = min(A);
        b1 = max(A);
        if a1 == b1
           A1 = 0.5;
        else
           A1 = (A-a1).*0.98./(b1-a1)+0.01;
        end
        P_Z2(abs(P_Z1) ~= 9999) = A1;
        P_Z2(P_Z1 == 9999) = 0.99;
        P_Z2(P_Z1 == -9999) = 0.01;  
        P_ZZ(:,i) = P_Z2;
        n = n+num_single_fea;
    end
    %P_ZZ = 1*P_ZZ(:,1)+0.3*P_ZZ(:,2);%0.6*P_ZZ(:,2)+mean(P_ZZ,2);++0.2*P_ZZ(:,4)+0.5*P_ZZ(:,5)
    a1 = min(P_ZZ );
    b1 = max(P_ZZ );
    if a1 == b1
       P_ZZ = 0.5;
    else
       P_ZZ = (P_ZZ-a1).*0.98./(b1-a1)+0.01;
    end 
  %% run the GLAD
    num_labelers = num_fa;
    num_sps = spnum;
    imageIds = zeros(num_labelers,num_sps);
    for i=1:num_labelers
        for j=1:num_sps
            imageIds(i,j)=j;
        end
    end
    imageIds=imageIds(:);
    labelerIds=zeros(num_labelers,num_sps);
    for i=1:num_labelers
        for j=1:num_sps
            labelerIds(i,j)=i;
        end
    end
    labelerIds=labelerIds(:);
    candi_label= L';
    labels = candi_label(:);
    [imageStats, labelerStats ] = em (imageIds, labelerIds, labels, P_ZZ, Alpha, Beta);
    inferLabels = imageStats{2} ;
    infer_label = double(inferLabels >= 0.5);
    
    %% smoothness   
    weights = makeweights3(sp_feat,paras.theta,adjMatrix,num_fea);
    A1 = weights;  % the similarity matrix of the graph
    dd = sum(A1);
    Da1 = sparse(1:spnum,1:spnum,dd); 
    clear dd;
    La1 = Da1 - paras.alpha*A1; 
    U = sparse(1:spnum,1:spnum,inferLabels);
    I1 = eye(spnum);
    final_infer = ((inv(paras.gamma*La1+U)))'*infer_label;%eye(spnum)
    % final_infer = (final_infer-min(final_infer))/(max(final_infer)-min(final_infer));        
    % final_infer = uint8(255*final_infer);
    %% rewrite the fused saliency map
    fused_map3 = zeros(height,width);
    for i = 1:spnum
        fused_map3(spstats(i).PixelIdxList) = final_infer(i)+infer_label(i);
    end
    fused_map3 = (fused_map3 -min(fused_map3 (:)))/(max(fused_map3(:) )-min(fused_map3(:) ));  
    fused_map3 = uint8(fused_map3*255);

    
    imwrite(fused_map3, [res_path imgspathdir(ii).name]);
    clear fused_map3 infer_label inferLabels image imdata imsegs spdata P_Z1 P_Z2
     
end
    

























