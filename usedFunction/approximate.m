function imgs = approximate(imgs, scale)
%���ͼ��ĸ��ý���ֵ
for i=1:numel(imgs)
    h{i} = imresize(imgs{i}, scale, 'bicubic');
    l{i} = imresize(h{i}, 1/scale, 'bicubic');
    l{i} = imgs{i}-l{i};
    l{i} = imresize(l{i}, scale, 'bicubic');
    h{i} = h{i}+l{i};%��һ��
    
    l{i} = imresize(h{i}, 1/scale, 'bicubic');
    l{i} = imgs{i}-l{i};
    l{i} = imresize(l{i}, scale, 'bicubic');
    imgs{i} = h{i}+l{i};%�ڶ���
end
end