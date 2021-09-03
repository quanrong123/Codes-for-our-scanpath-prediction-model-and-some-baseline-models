function sp_feat = mergwfeat(spdata,num_sp)
    sp_feat = zeros(num_sp,24);
    sp_feat(:,1) = mat2gray(spdata.R);
    sp_feat(:,2) = mat2gray(spdata.G);
    sp_feat(:,3) = mat2gray(spdata.B);
    sp_feat(:,4) = mat2gray(spdata.L);
    sp_feat(:,5) = mat2gray(spdata.a);
    sp_feat(:,6) = mat2gray(spdata.b);
    sp_feat(:,7) = mat2gray(spdata.H);
    sp_feat(:,8) = mat2gray(spdata.S);
    sp_feat(:,9) = mat2gray(spdata.V);
    for i = 1:15
        sp_feat(:,i+9) = mat2gray(spdata.texture(i,:))';
    end
    
end