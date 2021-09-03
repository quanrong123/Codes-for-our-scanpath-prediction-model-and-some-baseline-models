function bm = binary_seg(sal, th)
    th = th * mean(mean(sal));
    bm = sal;
    bm(sal >= th) = 1;
    bm(sal < th) = 0; 







