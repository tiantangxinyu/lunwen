%使用重叠窗口获得图像的网格
function grid = sampling_grid(img_size, window, overlap, border, scale)

if nargin < 5
    scale = 1;
end

if nargin < 4
    border = [0 0];   
end

if nargin < 3
    overlap = [0 0];    
end

% 把所有的网格操作的参数放大
window = window * scale;
overlap = overlap * scale;
border = border * scale;

% 为重叠窗口创建网格
index = reshape(1:prod(img_size), img_size);
grid = index(1:window(1), 1:window(2)) - 1;

% 计算网格位移的偏移量
skip = window - overlap; 
offset = index(1+border(1):skip(1):img_size(1)-window(1)+1-border(1), ...
               1+border(2):skip(2):img_size(2)-window(2)+1-border(2));
offset = reshape(offset, [1 1 numel(offset)]);

% 最后计算网格
grid = repmat(grid, [1 1 numel(offset)]) + repmat(offset, [window 1]);
