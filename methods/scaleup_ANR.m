function [imgs, midres] = scaleup_ANR(conf, imgs)

% 超分辨率迭代过程
    fprintf('Scale-Up Our Function');
    midres =approximate(imgs, conf.upsample_factor);%迭代反投影
    for i = 1:numel(midres)
        features = collect(conf, {midres{i}}, conf.upsample_factor, conf.filters);
        features = double(features);

        % 使用块的字典和锚定投影矩阵进行图像重建
                
        features = conf.V_pca'*features;
        
        patches = zeros(size(conf.PP,1),size(features,2));
        blocksize = 50000; %如果内存不足，则可以减少块大小。
        if size(conf.pointslo,2) > 10000
            blocksize = 500;
        end
        if size(features,2) < blocksize
            %D = abs(pdist2(conf.pointslo',features','cosine'));
            D = abs(conf.pointslo'*features); %直接使用内积绝对值
            %D = conf.pointslo'*features; 
            
            [val idx] = max(D);            

            for l = 1:size(features,2)            
                patches(:,l) = conf.PPs{idx(l)} * features(:,l);
            end
        else            
            
            for b = 1:blocksize:size(features,2)
                if b+blocksize-1 > size(features,2)
                    %D = abs(pdist2(conf.pointslo',features(:,b:end)','cosine'));    
                    D = abs(conf.pointslo'*features(:,b:end));
                    %D = conf.pointslo'*features(:,b:end);

                else
                    %D = abs(pdist2(conf.pointslo',features(:,b:b+blocksize-1)','cosine'));
                    D = abs(conf.pointslo'*features(:,b:b+blocksize-1));     
                    %D = conf.pointslo'*features(:,b:b+blocksize-1); 
                                
                end
                [val idx] = max(D);            

                for l = 1:size(idx,2)
                    patches(:,b-1+l) = conf.PPs{idx(l)} * features(:,b-1+l);
                end
                
            end
        end
        
        % 把前面锅炉掉的低频加入到重建的图像块里面        
        patches = patches + collect(conf, {midres{i}}, conf.scale, {});
        
        % 按照切割的坐标进行贴合，获得完整图像
        img_size = size(imgs{i}) * conf.scale;
        grid = sampling_grid(img_size, ...
            conf.window, conf.overlap, conf.border, conf.scale);
        result = overlap_add(patches, img_size, grid);
        imgs{i} = result; 
        fprintf('.');
    end
fprintf('\n');
