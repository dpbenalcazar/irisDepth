function plywrite2(filename,faces,verts,varargin)
% plywrite(filename,faces,verts)
% Will write a face vertex mesh data in ply format.
% faces -> polygonal descriptions in terms of vertex indices
% verts -> list of vertex coordinate triplets
% faces and verts can be obtained by using the MATLAB isosurface function.
%
% plywrite(filename,faces,verts,rgb,normals)
% Will add color information.
% rgb -> optional list of integer RGB triplets per vertex
% normals -> optional list of float triplets per vertex
%
% A by-product of ongoing computational materials science research 
% at MINED@Gatech.(http://mined.gatech.edu/)
%
% Copyright (c) 2015, Ahmet Cecen and MINED@Gatech -  All rights reserved.
% 
% This verssion has been modified from the original to include surface
% normals by Daniel Benalcazar 2019.

% Read optional arguments
    if nargin >=4
        if class(varargin{1}) == "double" || class(varargin{1}) == "single"
            rgb = im2uint8(varargin{1});
        else
            rgb = uint8(varargin{1});
        end
        
        if nargin == 5
            normals = varargin{2};
        end
    end

% Create File
    fileID = fopen(filename,'w');
   
% Insert Header
    fprintf(fileID, ...
        ['ply\n', ...
        'format ascii 1.0\n', ...
        'element vertex %u\n', ...
        'property float32 x\n', ...
        'property float32 y\n', ...
        'property float32 z\n'], ...
        length(verts));

if nargin >= 4 % Colored Mesh
    fprintf(fileID, [...
        'property uchar red\n', ...
        'property uchar green\n', ...
        'property uchar blue\n']);
end   

if nargin == 5 % Mesh with Normals
    fprintf(fileID, [...
        'property float nx\n', ...
        'property float ny\n', ...
        'property float nz\n']);
end

% End Headder
    fprintf(fileID, [...
        'element face %u\n', ...
        'property list uint8 int32 vertex_indices\n', ...
        'end_header\n'], ...
        length(faces));

% Insert Vertices
    for i=1:length(verts)
        fprintf(fileID, ...
            ['%.6f ', '%.6f ', '%.6f '], ...
            verts(i,1),verts(i,2),verts(i,3));
        if nargin == 3
            fprintf(fileID, newline);
        end
        if nargin >= 4 % Insert Color
            fprintf(fileID, ...
            ['%u ', '%u ', '%u '], ...
            rgb(i,1), rgb(i,2), rgb(i,3));
        end
        if nargin == 4
            fprintf(fileID, newline);
        end
        if nargin == 5 % Insert Normals
            fprintf(fileID, ...
            ['%.6f ', '%.6f ', '%.6f \n'], ...
            normals(i,1), normals(i,2), normals(i,3));
        end
    end

% Insert Faces
    dlmwrite(filename,[size(faces,2)*ones(length(faces),1),faces-1],...
        '-append','delimiter',' ','precision',10);

end
