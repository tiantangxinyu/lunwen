% Anchored Neighborhood Regression优化方法
%
% 2018/5/1. 徐雪,华南理工大学软件学院毕业设计
%
%主要运行代码
clear;  
  
%包含需要使用的路径
p = pwd;
addpath usedFunction %调用的函数，主要是训练期间调用的

addpath(fullfile(p, '/methods'));  %用户回归的方法，本课题使用的是优化的ANR方法

addpath(fullfile(p, '/ksvdbox')) % K-SVD 字典的训练算法

addpath(fullfile(p, '/ompbox')) % 正交匹配追踪算法

%全局参数
imgscale = 1; %图像本身倍数

upscaling = 2; %这里使用的放大倍数，可以使x2, x3, x4...

input_dir = 'Set5'; %这里使用来自Set5数据集的数据进行测试
%input_dir = 'Set14'; % 也可以使用来自Set5数据集的数据进行测试

pattern = '*.bmp'; % 需要处理的图像格式，Set5和Set14都是'*.bmp'格式的图片

dict_sizes = [16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536];%可以选择的字典大小

neighbors = [1:1:12, 16:4:32, 40:8:64, 80:16:128, 256, 512, 1024];%可以选择，从而进行投影矩阵计算的邻域大小
%d = 7
%for nn=1:28
%nn= 28

disp('The experiment corresponds to the results from Table 2 in the reference paper.');

disp(['The experiment uses ' input_dir ' dataset and aims at a magnification of factor x' num2str(upscaling) '.']);
if flag==1
    disp('All methods are employed : Bicubic, Yang et al., Zeyde et al., GR, ANR, NE+LS, NE+NNLS, and NE+LLE.');    
else
    disp('We run only for Bicubic, GR and ANR methods, the other get the Bicubic result by default.');
end

fprintf('\n\n');

for d=7%采用1024原子大小的字典    
    tag = [input_dir '_x' num2str(upscaling) '_' num2str(dict_sizes(d)) 'atoms'];
    
    disp(['使用放大因子x' num2str(upscaling) ' ' input_dir ' 的稀疏字典，大小为' num2str(dict_sizes(d))]);
    
    mat_file = ['conf_Zeyde_' num2str(dict_sizes(d)) '_finalx' num2str(upscaling)];    
    
    if exist([mat_file '.mat'],'file')%如果字典已经被训练了，直接导入
        disp(['导入字典...' mat_file]);
        load(mat_file, 'conf');
    else%否则训练字典                            
        disp(['字典大小为 ' num2str(dict_sizes(d)) ' ，使用 Zeyde等人的方法训练...']);
        % 训练的参数配置
        conf.scale = upscaling; %放大倍数
        conf.level = 1; %执行等级
        conf.window = [3 3]; %低分辨率图像块大小
        conf.border = [1 1]; %需要忽略的图像边缘

        % 特征提取时使用的高通过滤器 (为低分辨率块定义，从而获取高频细节部分信息)
        conf.upsample_factor = upscaling; % 放大的倍数
        O = zeros(1, conf.upsample_factor-1);
        G = [1 O -1]; %梯度算子
        L = [1 O -2 O 1]/2; % Laplacian算子
        conf.filters = {G, G.', L, L.'}; % 二维版本，在竖直和水平方向上都要进行处理，也就是4个级联的过滤器
        conf.interpolate_kernel = 'bicubic';%本课题使用的基础核方法为双三次

        conf.overlap = [1 1]; %块之间重叠的部分
        if upscaling <= 2
            conf.overlap = [2 2];
        end
        
        startt = tic;%训练字典开始
        conf = learn_dict(conf, load_images(...            
            glob('CVPR08-SR/Data/Training', '*.bmp') ...
            ), dict_sizes(d)); %使用91幅图像进行训练 ，这里调用usedFunction里的learn_dict函数     
        conf.overlap = conf.window - [1 1]; % 用于更好的重建的全重叠    
        conf.trainingtime = toc(startt);
        toc(startt)
        
        save(mat_file, 'conf');  %存储稀疏字典的相关内容                     
            
    end%导入字典部分完成
            
    if dict_sizes(d) < 1024
        lambda = 0.01;
    elseif dict_sizes(d) < 2048
        lambda = 0.1;
    elseif dict_sizes(d) < 8192
        lambda = 1;
    else
        lambda = 5;
    end%定义学习率因为字典大小为1024，ambda = 0.1
           
    if dict_sizes(d) < 10000
        conf.ProjM = inv(conf.dict_lores'*conf.dict_lores+lambda*eye(size(conf.dict_lores,2)))*conf.dict_lores';    
        conf.PP = (1+lambda)*conf.dict_hires*conf.ProjM;
    else
        % 这里只是近似
        conf.PP = zeros(size(conf.dict_hires,1), size(conf.V_pca,2));
        conf.ProjM = [];
    end
    
    conf.filenames = glob(input_dir, pattern); %这里是Set5数据集下的图片名
    
    conf.desc = {'Original', 'Bicubic', 'Our Function'};
    conf.results = {};
    
    %选取锚定点
    %conf.points = [1:10:size(conf.dict_lores,2)];
    conf.points = [1:1:size(conf.dict_lores,2)];
    conf.pointslo = conf.dict_lores(:,conf.points);
    conf.pointsloPCA = conf.pointslo'*conf.V_pca';
    
    %对于字典的锚定邻域和投影矩阵的预计算  
    conf.PPs = [];    
    if  size(conf.dict_lores,2) < 40
        clustersz = size(conf.dict_lores,2);
    else
        clustersz = 40;
    end

    %D = abs(conf.pointslo'*conf.dict_lores);
    %使用反向不变的余弦相似度绝对值作为度量
    D = conf.pointslo'*conf.dict_lores;
    for i=1:1024
        nor=norm(conf.dict_lores(:,i))*norm(conf.pointslo(:,i));
        D(i,:)=abs(D(i,:)/nor);
    end
    
    for i = 1:length(conf.points) 
        [vals idx] = sort(D(i,:), 'descend');%按照余弦相似性排序
        if (clustersz >= size(conf.dict_lores,2)/2)
            conf.PPs{i} = conf.PP;
        else
            Lo = conf.dict_lores(:, idx(1:clustersz));        
            conf.PPs{i} = 1.01*conf.dict_hires(:,idx(1:clustersz))*inv(Lo'*Lo+0.01*eye(size(Lo,2)))*Lo';    
        end%获得投影矩阵
    end
    
    save([tag '_' mat_file '_ANR_projections_imgscale_' num2str(imgscale)],'conf');%存储投影矩阵
    
    conf.result_dirImages = qmkdir([input_dir '/results_' tag]);%在Set5中存储灰阶图
    conf.result_dirImagesRGB = qmkdir([input_dir '/results_' tag 'RGB']);%在Set5中存储彩色图
    conf.result_dir = qmkdir(['Results-' datestr(now, 'YYYY-mm-dd_HH-MM-SS')]);%在文件夹中存储本次得到的灰度图
    conf.result_dirRGB = qmkdir(['ResultsRGB-' datestr(now, 'YYYY-mm-dd_HH-MM-SS')]);%在文件夹中存储本次得到的彩色图
    
    %%
    %测试阶段
    t = cputime;    
        
    conf.countedtime = zeros(numel(conf.desc),numel(conf.filenames));
    
    res =[];
    for i = 1:numel(conf.filenames)
        f = conf.filenames{i};
        [p, n, x] = fileparts(f);
        [img, imgCB, imgCR] = load_images({f}); 
        if imgscale<1
            img = resize(img, imgscale, conf.interpolate_kernel);
            imgCB = resize(imgCB, imgscale, conf.interpolate_kernel);
            imgCR = resize(imgCR, imgscale, conf.interpolate_kernel);
        end
        sz = size(img{1});
        
        fprintf('%d/%d\t"%s" [%d x %d]\n', i, numel(conf.filenames), f, sz(1), sz(2));
    
        img = modcrop(img, conf.scale^conf.level);
        imgCB = modcrop(imgCB, conf.scale^conf.level);
        imgCR = modcrop(imgCR, conf.scale^conf.level);

            low = resize(img, 1/conf.scale^conf.level, conf.interpolate_kernel);
            if ~isempty(imgCB{1})
                lowCB = resize(imgCB, 1/conf.scale^conf.level, conf.interpolate_kernel);
                lowCR = resize(imgCR, 1/conf.scale^conf.level, conf.interpolate_kernel);
            end
            
        interpolated = resize(low, conf.scale^conf.level, conf.interpolate_kernel);
        if ~isempty(imgCB{1})
            interpolatedCB = resize(lowCB, conf.scale^conf.level, conf.interpolate_kernel);    
            interpolatedCR = resize(lowCR, conf.scale^conf.level, conf.interpolate_kernel);    
        end
        
        res{1} = interpolated;
                               
        startt = tic;
        res{2} = scaleup_ANR(conf, low);
        toc(startt)
        conf.countedtime(2,i) = toc(startt);    
             
        result = cat(3, img{1}, interpolated{1}, res{2}{1});
        result = shave(uint8(result * 255), conf.border * conf.scale);
        
        if ~isempty(imgCB{1})
            resultCB = interpolatedCB{1};
            resultCR = interpolatedCR{1};           
            resultCB = shave(uint8(resultCB * 255), conf.border * conf.scale);
            resultCR = shave(uint8(resultCR * 255), conf.border * conf.scale);
        end

        conf.results{i} = {};
        for j = 1:numel(conf.desc)            
            conf.results{i}{j} = fullfile(conf.result_dirImages, [n sprintf('[%d-%s]', j, conf.desc{j}) x]);            
            imwrite(result(:, :, j), conf.results{i}{j});

            conf.resultsRGB{i}{j} = fullfile(conf.result_dirImagesRGB, [n sprintf('[%d-%s]', j, conf.desc{j}) x]);
            if ~isempty(imgCB{1})
                rgbImg = cat(3,result(:,:,j),resultCB,resultCR);
                rgbImg = ycbcr2rgb(rgbImg);
            else
                rgbImg = cat(3,result(:,:,j),result(:,:,j),result(:,:,j));
            end
            
            imwrite(rgbImg, conf.resultsRGB{i}{j});
        end        
        conf.filenames{i} = f;
    end   
    conf.duration = cputime - t;

    % 测试的性能
    scores = run_comparison(conf);
    process_scores_Tex(conf, scores,length(conf.filenames));
    
    run_comparisonRGB(conf); % 提供彩色图片和HTML结果对比
    %%    
    save([tag '_' mat_file '_results_imgscale_' num2str(imgscale)],'conf');%存储结果
end
%