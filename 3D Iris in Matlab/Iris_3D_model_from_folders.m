% This scripts helps obtaining the iris 3D model from all the images in a folder.

% For an example with synthetic iris images use:
% fold_rgb = '../datasets/micro_test/SYN-256x256/';
% fold_dep = '../datasets/micro_test/DEP-256x256/';

% For an example with translated images use:
fold_rgb = '../datasets/micro_test/S2R-256x256/';
fold_dep = '../datasets/micro_test/DEP-256x256/';

% Output folder:
fold_m3d = 'results/micro_test/';
mkdir(fold_m3d);

% Define scales along XY plane and Z axis:
XYscale = 13.4737/256;
Zscale = 1.9355;

% Read File Names:
Files_rgb = dir([fold_rgb, '*g']);
Files_dep = dir([fold_dep, '*g']);
Nf = length(Files_rgb);

% Wait-bar:
w = 0; dw = 1/Nf;
wb_title = ['Completed Iris 3D Models: 0 of ', num2str(Nf)];
wb = waitbar(w, wb_title);

for f = 1:Nf
    % Read Image and Depthmap:
    img = imread([fold_rgb, Files_rgb(f).name]);
    dep = imread([fold_dep, Files_dep(f).name]);
    [H,W,~] = size(img);

    % Obtain the iris 3D model:
    [verts, colors, normals] = rgbd2mesh(img, dep, XYscale, Zscale);
    pc = pointCloud(verts, 'Color', colors, 'Normal', normals);

    % Save as 3D Point Cloud:
    ID = Files_rgb(f).name;
    ID(end-4:end) = [];
    file3 = [fold_m3d, ID, '.ply'];
    pcwrite(pc, file3);

    w = w + dw;
    wb_title = ['Completed Iris 3D Models: ',num2str(f), ' of ', num2str(Nf)];
    waitbar(w, wb, wb_title);
end
close(wb)
