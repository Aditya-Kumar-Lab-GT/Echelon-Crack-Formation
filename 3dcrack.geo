SetFactory("OpenCASCADE");
Mesh.MeshSizeMin = 1e-8;
Lx = 25;
Ly = 25;
Lz = 25;
h=.1;
CrackZ=12.5;

// Define the points for the cuboid
Point(1) = {0, 0, 0, 1.0};     
Point(2) = {Lx, 0, 0, 1.0};    
Point(3) = {Lx, Ly, 0, 1.0};   
Point(4) = {0, Ly, 0, 1.0};     
Point(5) = {0, 0, Lz, 1.0};     
Point(6) = {Lx, 0, Lz, 1.0};    
Point(7) = {Lx, Ly, Lz, 1.0};   
Point(8) = {0, Ly, Lz, 1.0};   

Point(9) = {0, Ly/2-h/2, Lz, 1.0};     
Point(10) = {0, Ly/2+h/2, Lz, 1.0};    
Point(11) = {CrackZ, Ly/2, Lz, 1.0};  

Point(12) = {0, Ly/2-h/2, 0, 1.0};     
Point(13) = {0, Ly/2+h/2, 0, 1.0};    
Point(14) = {CrackZ, Ly/2, 0, 1.0};   
 

// Define lines for the cuboid
Line(1) = {1, 5};
Line(2) = {5,9};
Line(3) = {9,12};
Line(4) = {12,1};

Line(5) = {13,10};
Line(6) = {10,8};
Line(7) = {8,4};
Line(8) = {4,13};

Line(9) = {2, 6};
Line(10) = {6,7};
Line(11) = {7, 3};
Line(12) = {3,2};
//Until here vertical palnes

Line(13) = {5,6};
Line(14) = {8,7};

Line(15) = {1,2};
Line(16) = {4,3};

Line(17) = {9,11};
Line(18) = {11,14};
Line(19) = {14,12};

Line(20) = {10,11};
Line(21) = {14,13};
//Until here sleeping planes

// Define surfaces for the cuboid
Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Line Loop(2) = {5,6,7,8};
Plane Surface(2) = {2};
Line Loop(3) = {9,10,11,12};
Plane Surface(3) = {3};

Line Loop(4) = {13,-9,-15,1};
Plane Surface(4) = {4};
Line Loop(5) = {14,11,-16,-7};
Plane Surface(5) = {5};
Line Loop(6) = {17,18,19,-3};
Plane Surface(6) = {6};
Line Loop(7) = {20,18,21,5};
Plane Surface(7) = {7};

Line Loop(8) = {13,10,-14,-6,20,-17,-2};
Plane Surface(8) = {8};
Line Loop(9) = {15,-12,-16,8,-21,19,4};
Plane Surface(9) = {9};

// Define volume for the cuboid
Surface Loop(1) = {1, 2, 3, 4, 5, 6,7,8,9};
Volume(1) = {1};

Field[1] = Box;
Field[1].VIn = 0.1; //change this to .025
Field[1].VOut = 0.4;
Field[1].XMin = CrackZ-1;
Field[1].XMax = 16;
Field[1].YMin = Ly/2-2;
Field[1].YMax = Ly/2+2;
Field[1].ZMin = 0;
Field[1].ZMax = Lz;
Field[1].Thickness = 0.5;

Field[6] = Min;
Field[6].FieldsList = {1};
Background Field = 6;

// Mesh the structure
Mesh.ElementOrder = 1;
Mesh 3;

Mesh.MshFileVersion = 2.0;
Save "3dcrack.msh";//+
SetFactory("OpenCASCADE");
//+
SetFactory("OpenCASCADE");
//+
SetFactory("OpenCASCADE");
//+
SetFactory("Built-in");
//+
SetFactory("Built-in");
