# Generic imports
import os
import os.path
import PIL
import math
import scipy.special
import matplotlib
import pygmsh, meshio
import numpy             as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import qmc


# Custom imports
from meshes import *

### ************************************************
### Class defining shape object
class Shape:
    def __init__(self,
                 name          ='shape',
                 control_pts   =None,
                 n_control_pts =None,
                 n_sampling_pts=None,
                 radius        =None,
                 edgy          =None,
                 save_det_plot =None):
        if (name           is None): name           = 'shape'
        if (control_pts    is None): control_pts    = np.array([])
        if (n_control_pts  is None): n_control_pts  = 0
        if (n_sampling_pts is None): n_sampling_pts = 0
        if (radius         is None): radius         = np.array([])
        if (edgy           is None): edgy           = np.array([])
        if (save_det_plot  is None): save_det_plot  = True


        self.name           = name
        self.control_pts    = control_pts
        self.n_control_pts  = n_control_pts
        self.n_sampling_pts = n_sampling_pts
        self.curve_pts      = np.array([])
        self.curve_cpts     = np.array([])
        self.area           = 0.0
        self.size_x         = 0.0
        self.size_y         = 0.0
        self.index          = 0
        self.save_det_plot = save_det_plot
        self.bitmap = 0
        self.curve_length = 0

        if (len(radius) == n_control_pts): self.radius = radius
        if (len(radius) == 1):             self.radius = radius*np.ones([n_control_pts])

        if (len(edgy) == n_control_pts):   self.edgy = edgy
        if (len(edgy) == 1):               self.edgy = edgy*np.ones([n_control_pts])

        subname             = name.split('_')
        if (len(subname) == 2): # name is of the form shape_?.xxx
            self.name       = subname[0]
            index           = subname[1].split('.')[0]
            self.index      = int(index)
        if (len(subname) >  2): # name contains several '_'
            print('Please do not use several "_" char in shape name')
            quit()

        if (len(control_pts) > 0):
            self.control_pts   = control_pts
            self.n_control_pts = len(control_pts)

    ### ************************************************
    ### Reset object
    def reset(self):
        self.name           = 'shape'
        self.control_pts    = np.array([])
        self.n_control_pts  = 0
        self.n_sampling_pts = 0
        self.radius         = np.array([])
        self.edgy           = np.array([])
        self.curve_pts      = np.array([])
        self.area           = 0.0

    ### ************************************************
    ### Generate shape
    def generate(self, *args, **kwargs):
        # Handle optional argument
        centering = kwargs.get('centering', True)
        #cylinder = True
        cylinder  = kwargs.get('cylinder',  False)
        magnify   = kwargs.get('magnify',   1.0)
        ccws      = kwargs.get('ccws',      True)
        latin_gen = kwargs.get('latin_gen', False)
        latin_gen_outer_space = kwargs.get('latin_gen_outer_space', False)

        # Generate random control points if empty
        if (len(self.control_pts) == 0):
            if (cylinder):
                self.control_pts = generate_cylinder_pts(self.n_control_pts)
            elif (latin_gen):
                self.control_pts = latin_gen_outer_space[self.index,:].reshape(-1,2)
            else:
                self.control_pts = generate_random_pts(self.n_control_pts)
                

        # Magnify
        self.control_pts *= magnify

        # Center set of points
        center = np.mean(self.control_pts, axis=0)
        if (centering):
            self.control_pts -= center

        # Sort points counter-clockwise
        if (ccws):
            control_pts, radius, edgy  = ccw_sort(self.control_pts,
                                                  self.radius,
                                                  self.edgy)
        else:
            control_pts = np.array(self.control_pts)
            radius      = np.array(self.radius)
            edgy        = np.array(self.edgy)

        local_curves = []
        cont_pts = []
        delta        = np.zeros([self.n_control_pts,2])
        radii        = np.zeros([self.n_control_pts,2])
        delta_b      = np.zeros([self.n_control_pts,2])

        # Compute all informations to generate curves
        for i in range(self.n_control_pts):
            # Collect points
            prv  = (i-1)
            crt  = i
            nxt  = (i+1)%self.n_control_pts

            pt_m = control_pts[prv,:]
            pt_c = control_pts[crt,:]
            pt_p = control_pts[nxt,:]

            # Compute delta vector
            diff         = pt_p - pt_m
            delta[crt,:] = diff/np.linalg.norm(diff)

            # Compute edgy vector
            delta_b[crt,:] = 0.5*(pt_m + pt_p) - pt_c

            # Compute radii
            dist         = compute_distance(pt_m, pt_c)
            radii[crt,0] = 0.5*dist*radius[crt]

            dist         = compute_distance(pt_c, pt_p)
            radii[crt,1] = 0.5*dist*radius[crt]

        # Generate curves
        for i in range(self.n_control_pts):
            crt  = i
            nxt  = (i+1)%self.n_control_pts
            pt_c = control_pts[crt,:]
            pt_p = control_pts[nxt,:]
            dist = compute_distance(pt_c, pt_p)
            smpl = math.ceil(self.n_sampling_pts*math.sqrt(dist))

            local_curve,control_pt_out = generate_bezier_curve(pt_c,           pt_p,
                                                delta[crt,:],   delta[nxt,:],
                                                delta_b[crt,:], delta_b[nxt,:],
                                                radii[crt,1],   radii[nxt,0],
                                                edgy[crt],      edgy[nxt],
                                                smpl)
            
            local_curves.append(local_curve)
            cont_pts.append(control_pt_out)

        curve          = np.concatenate([c for c in local_curves])
        x, y           = curve.T
        z              = np.zeros(x.size)
        self.curve_pts = np.column_stack((x,y,z))
        self.curve_pts = remove_duplicate_pts(self.curve_pts)

        if self.save_det_plot == True:
            curve          = np.concatenate([c for c in cont_pts])
            x, y           = curve.T
            z              = np.zeros(x.size)
            self.curve_cpts = np.column_stack((x,y,z))

            # Center set of points
            if (centering):
                center            = np.mean(self.curve_pts, axis=0)
                self.curve_pts   -= center
                self.control_pts[:,0:2] -= center[0:2]
                self.curve_cpts -= center
            
            #plt.plot(self.curve_pts[:,0], self.curve_pts[:,1])
            colors = plt.cm.tab10  # or any other colormap
            for i in range(self.n_control_pts):
                curve          = local_curves[i]
                x, y           = curve.T
                z              = np.zeros(x.size)
                curve_pts = np.column_stack((x,y,z)) - center

                control_for_i = self.curve_cpts[i*4:i*4+4]

                x = np.cos(np.linspace(0,2*np.pi,100))*radii[i,1]
                y = np.sin(np.linspace(0,2*np.pi,100))*radii[i,1]
                color = colors(i % 10)
                plt.plot(x + control_pts[i, 0]- center[0], y + control_pts[i, 1]- center[1], color=color)
                
                """
                nxt = (i+1)%self.n_control_pts
                x = np.cos(np.linspace(0,2*np.pi,100))*radii[nxt,0]
                y = np.sin(np.linspace(0,2*np.pi,100))*radii[nxt,0]
                color = colors(i % 10)
                plt.plot(x + self.control_pts[nxt,0], y + self.control_pts[nxt, 1], color=color)
                """
                plt.plot(curve_pts[:, 0], curve_pts[:, 1], color=color)
                plt.scatter(control_for_i[1:3, 0], control_for_i[1:3, 1], color=color)
            filename = self.name+'_'+str(self.index)+"det_plot"+'.png'
            plt.savefig(filename,
                        dpi=400,
                        bbox_inches='tight')
            plt.clf() 


        # Compute area
        self.compute_area()

        # Compute dimensions
        self.compute_dimensions()

        # Compute the lenght of the curve
        self.compute_curve_length()

    ### ************************************************
    ### Write image
    def generate_image(self, *args, **kwargs):
        # Handle optional argument
        plot_pts       = kwargs.get('plot_pts',       False)
        override_name  = kwargs.get('override_name',  '')
        show_quadrants = kwargs.get('show_quadrants', False)
        max_radius     = kwargs.get('max_radius',     1.0)
        min_radius     = kwargs.get('min_radius',     0.2)
        xmin           = kwargs.get('xmin',          -1.0)
        xmax           = kwargs.get('xmax',           1.0)
        ymin           = kwargs.get('ymin',          -1.0)
        ymax           = kwargs.get('ymax',           1.0)
        # Plot shape
        plt.xlim([xmin,xmax])
        plt.ylim([ymin,ymax])
        plt.axis('off')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.fill([xmin,xmax,xmax,xmin],
                 [ymin,ymin,ymax,ymax],
                 color=(0.784,0.773,0.741),
                 linewidth=2.5,
                 zorder=0)
        plt.fill(self.curve_pts[:,0],
                 self.curve_pts[:,1],
                 'black',
                 linewidth=0,
                 zorder=1)

        # Plot points
        # Each point gets a different color
        if (plot_pts):
            colors = matplotlib.cm.ocean(np.linspace(0, 1, self.n_control_pts))
            plt.scatter(self.control_pts[:,0],
                        self.control_pts[:,1],
                        color=colors,
                        s=50,
                        zorder=2,
                        alpha=1.0)

        # Plot quadrants
        if (show_quadrants):
            for pt in range(self.n_control_pts):
                dangle = (360.0/float(self.n_control_pts))
                angle  = dangle*float(pt)+dangle/2.0
                x_max  = max_radius*math.cos(math.radians(angle))
                y_max  = max_radius*math.sin(math.radians(angle))
                x_min  = min_radius*math.cos(math.radians(angle))
                y_min  = min_radius*math.sin(math.radians(angle))
                plt.plot([x_min, x_max],
                         [y_min, y_max],
                         color='w',
                         linewidth=1)

            circle = plt.Circle((0,0),max_radius,fill=False,color='w')
            plt.gcf().gca().add_artist(circle)
            circle = plt.Circle((0,0),min_radius,fill=False,color='w')
            plt.gcf().gca().add_artist(circle)

        # Save image
        filename = self.name+'_'+str(self.index)+'.png'
        if (override_name != ''): filename = override_name

        plt.gcf().tight_layout()
        plt.savefig(filename,
                    dpi=400,
                    bbox_inches='tight')
        plt.close(plt.gcf())
        plt.cla()
        trim_white(filename)

    def generate_bitmap(self,bit_im = False,  *args, **kwargs):
        # Handle optional argument
        plot_pts       = self.curve_pts
        bitmap = scanline_fill(plot_pts[:,:2] *250 + np.array([300,300]), 600,600)
        self.bitmap = bitmap
        if bit_im:
            plt.imshow(bitmap, cmap='gray')
            # Save image
            filename = self.name+'_'+str(self.index)+"bit"+'.png'        
            plt.savefig(filename,
                        bbox_inches='tight')
            plt.clf()

        return bitmap

    ### ************************************************
    ### Write csv
    def write_csv(self):
        with open(self.name+'_'+str(self.index)+'.csv','w') as file:
            # Write header
            file.write('{} {}\n'.format(self.n_control_pts,
                                        self.n_sampling_pts))

            # Write control points coordinates
            for i in range(0,self.n_control_pts):
                file.write('{} {} {} {}\n'.format(self.control_pts[i,0],
                                                  self.control_pts[i,1],
                                                  self.radius[i],
                                                  self.edgy[i]))

    ### ************************************************
    ### Read csv and initialize shape with it
    def read_csv(self, filename, *args, **kwargs):
        # Handle optional argument
        keep_numbering = kwargs.get('keep_numbering', False)

        if (not os.path.isfile(filename)):
            print('I could not find csv file: '+filename)
            print('Exiting now')
            exit()

        self.reset()
        sfile  = filename.split('.')
        sfile  = sfile[-2]
        sfile  = sfile.split('/')
        name   = sfile[-1]

        if (keep_numbering):
            sname = name.split('_')
            name  = sname[0]
            name  = name+'_'+str(self.index)

        x      = []
        y      = []
        radius = []
        edgy   = []

        with open(filename) as file:
            header         = file.readline().split()
            n_control_pts  = int(header[0])
            n_sampling_pts = int(header[1])

            for i in range(0,n_control_pts):
                coords = file.readline().split()
                x.append(float(coords[0]))
                y.append(float(coords[1]))
                radius.append(float(coords[2]))
                edgy.append(float(coords[3]))
                control_pts = np.column_stack((x,y))

        self.__init__(name,
                      control_pts,
                      n_control_pts,
                      n_sampling_pts,
                      radius,
                      edgy)

    ### ************************************************
    ### Mesh shape
    def mesh(self, *args, **kwargs):
        # Handle optional argument
        mesh_domain = kwargs.get('mesh_domain', False)
        xmin        = kwargs.get('xmin',       -1.0)
        xmax        = kwargs.get('xmax',        1.0)
        ymin        = kwargs.get('ymin',       -1.0)
        ymax        = kwargs.get('ymax',        1.0)
        domain_h    = kwargs.get('domain_h',    1.0)
        mesh_format = kwargs.get('mesh_format', 'mesh')

        # Convert curve to polygon
        with pygmsh.geo.Geometry() as geom:
            poly      = geom.add_polygon(self.curve_pts,
                                         make_surface=not mesh_domain)

            # Mesh domain if necessary
            if (mesh_domain):
                # Compute an intermediate mesh size
                border = geom.add_rectangle(xmin, xmax,
                                            ymin, ymax,
                                            0.0,
                                            domain_h,
                                            holes=[poly.curve_loop])

            # Generate mesh and write in medit format
            try:
                mesh = geom.generate_mesh(dim=2)
            except AssertionError:
                print('\n'+'!!!!! Meshing failed !!!!!')
                return False, 0

        # Compute data from mesh
        n_tri = len(mesh.cells_dict['triangle'])

        # Remove vertex keywork from cells dictionnary
        # to avoid warning message from meshio
        del mesh.cells_dict['vertex']

        # Remove lines if output format is xml
        if (mesh_format == 'xml'): del mesh.cells['line']

        # Write mesh
        filename = self.name+'_'+str(self.index)+'.'+mesh_format
        meshio.write_points_cells(filename, mesh.points, mesh.cells)

        return True, n_tri

    ### ************************************************
    ### Get shape bounding box
    def compute_bounding_box(self):
        x_max, y_max = np.amax(self.control_pts,axis=0)
        x_min, y_min = np.amin(self.control_pts,axis=0)

        dx = x_max - x_min
        dy = y_max - y_min

        return dx, dy

    ### ************************************************
    ### Modify shape given a deformation field
    def modify_shape_from_field(self, deformation, *args, **kwargs):
        # Handle optional argument
        replace  = kwargs.get('replace',  False)
        pts_list = kwargs.get('pts_list', [])

        # Check inputs
        if (pts_list == []):
            if (len(deformation[:,0]) != self.n_control_pts):
                print('Input deformation field does not have right length')
                quit()
        if (len(deformation[0,:]) not in [2, 3]):
            print('Input deformation field does not have right width')
            quit()
        if (pts_list != []):
            if (len(pts_list) != len(deformation)):
                print('Lengths of pts_list and deformation are different')
                quit()

        # If shape is to be replaced entirely
        if (    replace):
            # If a list of points is provided
            if (pts_list != []):
                for i in range(len(pts_list)):
                    self.control_pts[pts_list[i],0] = deformation[i,0]
                    self.control_pts[pts_list[i],1] = deformation[i,1]
                    self.edgy[pts_list[i]]          = deformation[i,2]
            # Otherwise
            if (pts_list == []):
                self.control_pts[:,0] = deformation[:,0]
                self.control_pts[:,1] = deformation[:,1]
                self.edgy[:]          = deformation[:,2]
        # Otherwise
        if (not replace):
            # If a list of points to deform is provided
            if (pts_list != []):
                for i in range(len(pts_list)):
                    self.control_pts[pts_list[i],0] += deformation[i,0]
                    self.control_pts[pts_list[i],1] += deformation[i,1]
                    self.edgy[pts_list[i]]          += deformation[i,2]
            # Otherwise
            if (pts_list == []):
                self.control_pts[:,0] += deformation[:,0]
                self.control_pts[:,1] += deformation[:,1]
                self.edgy[:]          += deformation[:,2]

        # Increment shape index
        self.index += 1

    ### ************************************************
    ### Compute shape area
    def compute_curve_length(self):
        curve = self.curve_pts
        diffs = np.diff(curve, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        self.curve_length = np.sum(segment_lengths)

    def compute_area(self):
        self.area = 0.0

        # Use Green theorem to compute area
        for i in range(0,len(self.curve_pts)-1):
            x1 = self.curve_pts[i-1,0]
            x2 = self.curve_pts[i,  0]
            y1 = self.curve_pts[i-1,1]
            y2 = self.curve_pts[i,  1]

            self.area += 2.0*(y1+y2)*(x2-x1)

    ### ************************************************
    ### Compute shape dimensions
    def compute_dimensions(self):
        self.size_y = 0.0
        self.size_x = 0.0
        xmin = 1.0e20
        xmax =-1.0e20
        ymin = 1.0e20
        ymax =-1.0e20

        for i in range(len(self.curve_pts)):
            xmin = min(xmin, self.curve_pts[i,0])
            xmax = max(xmax, self.curve_pts[i,0])
            ymin = min(ymin, self.curve_pts[i,1])
            ymax = max(ymax, self.curve_pts[i,1])

        self.size_x = xmax - xmin
        self.size_y = ymax - ymin

### End of class Shape
### ************************************************

### ************************************************
### Compute distance between two points
def compute_distance(p1, p2):

    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

### ************************************************
### Generate n_pts random points in the unit square
def generate_random_pts(n_pts):
    equ_dim_lim = 1.5

    latin_gen = qmc.LatinHypercube(d=2)
    latin_gen = latin_gen.random(n=n_pts)
    latin_gen = equ_dim_lim * latin_gen
    return latin_gen

    #return np.random.randn(n_pts,2) * equ_dim_lim

### ************************************************
### Generate cylinder points
def generate_cylinder_pts(n_pts):
    if (n_pts < 4):
        print('Not enough points to generate cylinder')
        exit()

    pts = np.zeros([n_pts, 2])
    ang = 2.0*math.pi/n_pts
    for i in range(0,n_pts):
        pts[i,:] = [math.cos(float(i)*ang),math.sin(float(i)*ang)]

    return pts

### ************************************************
### Compute minimal distance between successive pts in array
def compute_min_distance(pts):
    dist_min = 1.0e20
    for i in range(len(pts)-1):
        p1       = pts[i  ,:]
        p2       = pts[i+1,:]
        dist     = compute_distance(p1,p2)
        dist_min = min(dist_min,dist)

    return dist_min

### ************************************************
### Remove duplicate points in input coordinates array
### WARNING : this routine is highly sub-optimal
def remove_duplicate_pts(pts):
    to_remove = []

    for i in range(len(pts)):
        for j in range(len(pts)):
            # Check that i and j are not identical
            if (i == j):
                continue

            # Check that i and j are not removed points
            if (i in to_remove) or (j in to_remove):
                continue

            # Compute distance between points
            pi = pts[i,:]
            pj = pts[j,:]
            dist = compute_distance(pi,pj)

            # Tag the point to be removed
            if (dist < 1.0e-8):
                to_remove.append(j)

    # Sort elements to remove in reverse order
    to_remove.sort(reverse=True)

    # Remove elements from pts
    for pt in to_remove:
        pts = np.delete(pts, pt, 0)

    return pts

### ************************************************
### Counter Clock-Wise sort
###  - Take a cloud of points and compute its geometric center
###  - Translate points to have their geometric center at origin
###  - Compute the angle from origin for each point
###  - Sort angles by ascending order
def ccw_sort(pts, rad, edg):
    geometric_center = np.mean(pts,axis=0)
    translated_pts   = pts - geometric_center
    angles           = np.arctan2(translated_pts[:,1], translated_pts[:,0])
    x                = angles.argsort()
    pts2             = np.array(pts)
    rad2             = np.array(rad)
    edg2             = np.array(edg)

    return pts2[x,:], rad2[x], edg2[x]

### ************************************************
### Compute Bernstein polynomial value
def compute_bernstein(n,k,t):
    k_choose_n = scipy.special.binom(n,k)

    return k_choose_n * (t**k) * ((1.0-t)**(n-k))

### ************************************************
### Sample Bezier curves given set of control points
### and the number of sampling points
### Bezier curves are parameterized with t in [0,1]
### and are defined with n control points P_i :
### B(t) = sum_{i=0,n} B_i^n(t) * P_i
def sample_bezier_curve(control_pts, n_sampling_pts):
    n_control_pts = len(control_pts)
    t             = np.linspace(0, 1, n_sampling_pts)
    curve         = np.zeros((n_sampling_pts, 2))

    for i in range(n_control_pts):
        curve += np.outer(compute_bernstein(n_control_pts-1, i, t),
                          control_pts[i])

    return curve

### ************************************************
### Generate Bezier curve between two pts
def generate_bezier_curve(p1,       p2,
                          delta1,   delta2,
                          delta_b1, delta_b2,
                          radius1,  radius2,
                          edgy1,    edgy2,
                          n_sampling_pts):

    # Sample the curve if necessary
    if (n_sampling_pts != 0):
        # Create array of control pts for cubic Bezier curve
        # First and last points are given, while the two intermediate
        # points are computed from edge points, angles and radius
        control_pts      = np.zeros((4,2))
        control_pts[0,:] = p1[:]
        control_pts[3,:] = p2[:]

        # Compute baseline intermediate control pts ctrl_p1 and ctrl_p2
        ctrl_p1_base = radius1*delta1
        ctrl_p2_base =-radius2*delta2

        ctrl_p1_edgy = delta_b1
        ctrl_p2_edgy = delta_b2

        control_pts[1,:] = p1 + ctrl_p1_base + edgy1*ctrl_p1_edgy
        control_pts[2,:] = p2 + ctrl_p2_base + edgy2*ctrl_p2_edgy

        # Compute points on the Bezier curve
        curve = sample_bezier_curve(control_pts, n_sampling_pts)

    # Else return just a straight line
    else:
        curve = p1
        curve = np.vstack([curve,p2])

    return curve, control_pts

### Crop white background from image
def trim_white(filename):

    im   = PIL.Image.open(filename)
    bg   = PIL.Image.new(im.mode, im.size, (255,255,255))
    diff = PIL.ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    cp   = im.crop(bbox)
    cp.save(filename)


def scanline_fill(polygon, width, height):
    # Convert to integer pixel coordinates
    polygon = np.array(polygon, dtype=int)
    filled = np.zeros((height, width), dtype=np.uint8)

    ymin = np.min(polygon[:,1])
    ymax = np.max(polygon[:,1])

    for y in range(ymin, ymax + 1):
        intersections = []
        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]
            if p1[1] == p2[1]: continue  # skip horizontal edges
            if (y < min(p1[1], p2[1])) or (y > max(p1[1], p2[1])): continue
            # Compute intersection x
            x = int(p1[0] + (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]))
            intersections.append(x)
        intersections.sort()
        unique_list = []
        seen = set()
        for item in intersections:
            if item not in seen:
                seen.add(item)
                unique_list.append(item)
        intersections = unique_list
        for i in range(0, len(intersections), 2):
            if i + 1 < len(intersections):
                x_start = intersections[i]
                x_end = intersections[i + 1]
                filled[y, x_start:x_end+1] = 1
    return filled