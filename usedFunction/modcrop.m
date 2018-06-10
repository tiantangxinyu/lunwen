function imgs = modcrop(imgs, modulo)
%剪裁操作，使得图像和放大倍数为整数倍
for i = 1:numel(imgs)
    sz = size(imgs{i});
    sz = sz - mod(sz, modulo);
    imgs{i} = imgs{i}(1:sz(1), 1:sz(2));
end
