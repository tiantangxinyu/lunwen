function imgs = approximate(imgs, scale)
%获得图像的更好近似值
for i=1:numel(imgs)
    h{i} = imresize(imgs{i}, scale, 'bicubic');
    l{i} = imresize(h{i}, 1/scale, 'bicubic');
    l{i} = imgs{i}-l{i};
    l{i} = imresize(l{i}, scale, 'bicubic');
    h{i} = h{i}+l{i};%第一次
    
    l{i} = imresize(h{i}, 1/scale, 'bicubic');
    l{i} = imgs{i}-l{i};
    l{i} = imresize(l{i}, scale, 'bicubic');
    imgs{i} = h{i}+l{i};%第二次
end
end