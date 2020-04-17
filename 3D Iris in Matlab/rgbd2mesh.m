function [pts, colors, normals, varargout] = rgbd2mesh(image, depthmap, varargin)
% [pts, colors, normals] = rgbd2mesh(image, depthmap) 
% Produces the 3D pointcloud model from the imput image and depthmap,
% where pts are the coordinates of the 3D points, colors stores the color 
% of each 3D point, and normals contains the estimated normal of the best 
% fitting plane arround each 3D point, using 6 neighbors. 
%
% [pts, colors, normals] = rgbd2mesh(image, depthmap, XYscale, Zscale) 
% XYscale and Zscale changes the scale values arround the XY plane and the 
% Z axis correspondingly. The deffoult values are: XYscale=1, Zscale=30.
%
% [verts, colors, normals, faces] = rgbd2mesh(image, depthmap, __ ) 
% Produces the mesh model, where verts are the vertices of the mesh, and
% faces holds the connectivity of the vertices. The vertices are conected
% using sqares from the neighboring pixels in the original image.
%
% This code was produced to create 3D models of the human iris. To see CNN
% implementation of this method please visit: 
% https://github.com/dpbenalcazar/irisDepth
%
% Author: Daniel Benalcazar 2019


    % Obtain scale values in the XY plane, as well as the Z axis:
    if nargin <= 2
        XYsc = 1;
        Zsc = 30;
    else
        XYsc = varargin{1};
        Zsc = varargin{2};
    end
    
    % Change image and depthmap to double precision floats:
    if class(image) ~= "double"
        image = im2double(image);
    end
    if class(depthmap) ~= "double"
        depthmap = im2double(depthmap);
    end
    
    % Create mesh grid in the XY plane:
    [H,W,~] = size(image);
    im2 = flipud(image);
    [X,Y] = meshgrid(1:W,1:H);
    X = XYsc * (X - W/2);
    Y = XYsc * (Y - H/2);

    % Normalize depthmap and obtain the values in theZ axis:
    depthmap = imresize(depthmap,[H,W]);
    z = 1 - depthmap;
    z = z - min(z(:));
    Z = Zsc*flipud(z);
    
    % Obtain vector coordinates of 3D points:
    if nargout == 4 
        % Compute connectivity only if necesary
        [faces, pts, ~] = surf2patch(X,Y,Z); 
        varargout{1} = faces;
    else
        pts = [X(:), Y(:), Z(:)];
    end
    
    % Obtain color information from RGB image
    R = im2(:,:,1);
    G = im2(:,:,2);
    B = im2(:,:,3);
    colors = [R(:), G(:), B(:)];
    
    % Calculate normals from 3D points:
    pc = pointCloud(pts); 
    normals = pcnormals(pc);
    
end

