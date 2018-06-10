% Anchored Neighborhood Regression�Ż�����
%
% 2018/5/1. ��ѩ,��������ѧ���ѧԺ��ҵ���
%
%��Ҫ���д���
clear;  
  
%������Ҫʹ�õ�·��
p = pwd;
addpath usedFunction %���õĺ�������Ҫ��ѵ���ڼ���õ�

addpath(fullfile(p, '/methods'));  %�û��ع�ķ�����������ʹ�õ����Ż���ANR����

addpath(fullfile(p, '/ksvdbox')) % K-SVD �ֵ��ѵ���㷨

addpath(fullfile(p, '/ompbox')) % ����ƥ��׷���㷨

%ȫ�ֲ���
imgscale = 1; %ͼ������

upscaling = 2; %����ʹ�õķŴ���������ʹx2, x3, x4...

input_dir = 'Set5'; %����ʹ������Set5���ݼ������ݽ��в���
%input_dir = 'Set14'; % Ҳ����ʹ������Set5���ݼ������ݽ��в���

pattern = '*.bmp'; % ��Ҫ�����ͼ���ʽ��Set5��Set14����'*.bmp'��ʽ��ͼƬ

dict_sizes = [16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536];%����ѡ����ֵ��С

neighbors = [1:1:12, 16:4:32, 40:8:64, 80:16:128, 256, 512, 1024];%����ѡ�񣬴Ӷ�����ͶӰ�������������С
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

for d=7%����1024ԭ�Ӵ�С���ֵ�    
    tag = [input_dir '_x' num2str(upscaling) '_' num2str(dict_sizes(d)) 'atoms'];
    
    disp(['ʹ�÷Ŵ�����x' num2str(upscaling) ' ' input_dir ' ��ϡ���ֵ䣬��СΪ' num2str(dict_sizes(d))]);
    
    mat_file = ['conf_Zeyde_' num2str(dict_sizes(d)) '_finalx' num2str(upscaling)];    
    
    if exist([mat_file '.mat'],'file')%����ֵ��Ѿ���ѵ���ˣ�ֱ�ӵ���
        disp(['�����ֵ�...' mat_file]);
        load(mat_file, 'conf');
    else%����ѵ���ֵ�                            
        disp(['�ֵ��СΪ ' num2str(dict_sizes(d)) ' ��ʹ�� Zeyde���˵ķ���ѵ��...']);
        % ѵ���Ĳ�������
        conf.scale = upscaling; %�Ŵ���
        conf.level = 1; %ִ�еȼ�
        conf.window = [3 3]; %�ͷֱ���ͼ����С
        conf.border = [1 1]; %��Ҫ���Ե�ͼ���Ե

        % ������ȡʱʹ�õĸ�ͨ������ (Ϊ�ͷֱ��ʿ鶨�壬�Ӷ���ȡ��Ƶϸ�ڲ�����Ϣ)
        conf.upsample_factor = upscaling; % �Ŵ�ı���
        O = zeros(1, conf.upsample_factor-1);
        G = [1 O -1]; %�ݶ�����
        L = [1 O -2 O 1]/2; % Laplacian����
        conf.filters = {G, G.', L, L.'}; % ��ά�汾������ֱ��ˮƽ�����϶�Ҫ���д���Ҳ����4�������Ĺ�����
        conf.interpolate_kernel = 'bicubic';%������ʹ�õĻ����˷���Ϊ˫����

        conf.overlap = [1 1]; %��֮���ص��Ĳ���
        if upscaling <= 2
            conf.overlap = [2 2];
        end
        
        startt = tic;%ѵ���ֵ俪ʼ
        conf = learn_dict(conf, load_images(...            
            glob('CVPR08-SR/Data/Training', '*.bmp') ...
            ), dict_sizes(d)); %ʹ��91��ͼ�����ѵ�� ���������usedFunction���learn_dict����     
        conf.overlap = conf.window - [1 1]; % ���ڸ��õ��ؽ���ȫ�ص�    
        conf.trainingtime = toc(startt);
        toc(startt)
        
        save(mat_file, 'conf');  %�洢ϡ���ֵ���������                     
            
    end%�����ֵ䲿�����
            
    if dict_sizes(d) < 1024
        lambda = 0.01;
    elseif dict_sizes(d) < 2048
        lambda = 0.1;
    elseif dict_sizes(d) < 8192
        lambda = 1;
    else
        lambda = 5;
    end%����ѧϰ����Ϊ�ֵ��СΪ1024��ambda = 0.1
           
    if dict_sizes(d) < 10000
        conf.ProjM = inv(conf.dict_lores'*conf.dict_lores+lambda*eye(size(conf.dict_lores,2)))*conf.dict_lores';    
        conf.PP = (1+lambda)*conf.dict_hires*conf.ProjM;
    else
        % ����ֻ�ǽ���
        conf.PP = zeros(size(conf.dict_hires,1), size(conf.V_pca,2));
        conf.ProjM = [];
    end
    
    conf.filenames = glob(input_dir, pattern); %������Set5���ݼ��µ�ͼƬ��
    
    conf.desc = {'Original', 'Bicubic', 'Our Function'};
    conf.results = {};
    
    %ѡȡê����
    %conf.points = [1:10:size(conf.dict_lores,2)];
    conf.points = [1:1:size(conf.dict_lores,2)];
    conf.pointslo = conf.dict_lores(:,conf.points);
    conf.pointsloPCA = conf.pointslo'*conf.V_pca';
    
    %�����ֵ��ê�������ͶӰ�����Ԥ����  
    conf.PPs = [];    
    if  size(conf.dict_lores,2) < 40
        clustersz = size(conf.dict_lores,2);
    else
        clustersz = 40;
    end

    %D = abs(conf.pointslo'*conf.dict_lores);
    %ʹ�÷��򲻱���������ƶȾ���ֵ��Ϊ����
    D = conf.pointslo'*conf.dict_lores;
    for i=1:1024
        nor=norm(conf.dict_lores(:,i))*norm(conf.pointslo(:,i));
        D(i,:)=abs(D(i,:)/nor);
    end
    
    for i = 1:length(conf.points) 
        [vals idx] = sort(D(i,:), 'descend');%������������������
        if (clustersz >= size(conf.dict_lores,2)/2)
            conf.PPs{i} = conf.PP;
        else
            Lo = conf.dict_lores(:, idx(1:clustersz));        
            conf.PPs{i} = 1.01*conf.dict_hires(:,idx(1:clustersz))*inv(Lo'*Lo+0.01*eye(size(Lo,2)))*Lo';    
        end%���ͶӰ����
    end
    
    save([tag '_' mat_file '_ANR_projections_imgscale_' num2str(imgscale)],'conf');%�洢ͶӰ����
    
    conf.result_dirImages = qmkdir([input_dir '/results_' tag]);%��Set5�д洢�ҽ�ͼ
    conf.result_dirImagesRGB = qmkdir([input_dir '/results_' tag 'RGB']);%��Set5�д洢��ɫͼ
    conf.result_dir = qmkdir(['Results-' datestr(now, 'YYYY-mm-dd_HH-MM-SS')]);%���ļ����д洢���εõ��ĻҶ�ͼ
    conf.result_dirRGB = qmkdir(['ResultsRGB-' datestr(now, 'YYYY-mm-dd_HH-MM-SS')]);%���ļ����д洢���εõ��Ĳ�ɫͼ
    
    %%
    %���Խ׶�
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

    % ���Ե�����
    scores = run_comparison(conf);
    process_scores_Tex(conf, scores,length(conf.filenames));
    
    run_comparisonRGB(conf); % �ṩ��ɫͼƬ��HTML����Ա�
    %%    
    save([tag '_' mat_file '_results_imgscale_' num2str(imgscale)],'conf');%�洢���
end
%