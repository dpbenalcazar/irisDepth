% This is an example of using an iris image and the corresponding depthmap
% to obtain the iris 3D model of the subject.

% For an example with synthetic iris images use:
% file1 = '../datasets/micro_test/SYN-256x256/Ren256x256_IT024_C2_E10_R088.png';
% file2 = '../datasets/micro_test/DEP-256x256/Dep256x256_IT024_E10_R088.png';

% For an example with translated images use:
file1 = '../datasets/micro_test/S2R-256x256/Ren256x256_IT024_C2_E10_R088_fake.png';
file2 = '../datasets/micro_test/DEP-256x256/Dep256x256_IT024_E10_R088.png';

% Output ID for saving the 3D model:
ID = 'S2R_IT024_C2_E10_R088'; % ID = 'SYN_IT024_C2_E10_R088';

% Read Image and Depthmap:
img = imread(file1);
dep = imread(file2);
[H,W,~] = size(img);

% Define scales along XY plane and Z axis:
XYscale = 13.4737/H;
Zscale = 1.9355;

% Obtain the iris 3D model:
[verts, colors, normals, faces] = rgbd2mesh(img, dep, XYscale, Zscale);

% Save as 3D Point Cloud:
pc = pointCloud(verts, 'Color', colors, 'Normal', normals);
file3 = ['results/',ID,'_ptcl.ply'];
pcwrite(pc, file3);

% Seve as 3D Mesh:
file4 = ['results/',ID,'_mesh.ply'];
plywrite2(file4, faces, verts, colors, normals);

% Show Point Cloud model:
pcshow(pc,'MarkerSize',20)
title(dash2space(ID));
drawnow;
