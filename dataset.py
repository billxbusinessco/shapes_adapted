# Generic imports
#%%
import os
import random
import shutil
import progress.bar
from   datetime import datetime
from scipy.stats import qmc
import pandas as pd


# Custom imports
from shapes import *
from meshes import *

### ************************************************
### Generate full dataset
# Parameters
n_sampling_pts = 20
mesh_domain    = False
plot_pts       = True
show_quadrants = True
n_shapes       = 10
time           = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
dataset_dir    = 'dataset_'+time+'/'
mesh_dir       = dataset_dir+'meshes/'
img_dir        = dataset_dir+'images/'
det_img_dir    = dataset_dir+'det_images/'
bit_img_dir    = dataset_dir+'bit_images/'


filename       = 'shape'
magnify        = 1.0
max_radius = (2)**0.5
xmin           =-2.0
xmax           = 2.0
ymin           =-2.0
ymax           = 2.0
n_tri_max      = 5000
n_pts = 4

### generate latin hypercube points in the defined space
equ_dim_lim = 1.5
latin_gen = False
latin_gen = qmc.LatinHypercube(d=n_pts*2)
latin_gen = latin_gen.random(n=n_shapes)
latin_gen = equ_dim_lim * latin_gen


# Create directories if necessary
if not os.path.exists(mesh_dir):
    os.makedirs(mesh_dir)
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
if not os.path.exists(det_img_dir):
    os.makedirs(det_img_dir)
if not os.path.exists(bit_img_dir):
    os.makedirs(bit_img_dir)

# Generate dataset
bar = progress.bar.Bar('Generating shapes', max=n_shapes)
df = pd.DataFrame(columns=['curve_points', 'area', 'perimeter','bitmap'])

for i in range(0,n_shapes):
    generated = False
    while (not generated):

        #n_pts  = random.randint(3, 7)
        radius = np.random.uniform(0.0, 1.0, size=n_pts)
        edgy   = np.random.uniform(0.0, .5, size=n_pts)
        shape  = Shape(filename+'_'+str(i),
                       None,
                       n_pts,
                       n_sampling_pts,
                       radius,
                       edgy,
                       save_det_plot = True)
        
        shape.generate(magnify=1.0,
                       xmin=xmin,
                       xmax=xmax,
                       ymin=ymin,
                       ymax=ymax,
                       latin_gen = latin_gen)

        img  = filename+'_'+str(i)+"det_plot"+'.png'
        shutil.move(img,  det_img_dir)

        shape.generate_bitmap()
        img  = filename+'_'+str(i)+"bit"+'.png'
        shutil.move(img,  bit_img_dir)

        new_rows = [{'curve_points': shape.curve_pts, 'area': shape.area, 'perimeter': shape.curve_length, "bitmap": shape.bitmap}]
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        generated = True
        #meshed, n_tri = shape.mesh()
        """
        if (True):
            shape.generate_image(plot_pts=plot_pts,
                                 max_radius = max_radius,
                                 xmin=xmin,
                                 xmax=xmax,
                                 ymin=ymin,
                                 ymax=ymax,
                                 show_quadrants=True)
            img  = filename+'_'+str(i)+'.png'
            mesh = filename+'_'+str(i)+'.mesh'
            shutil.move(img,  img_dir)
            #shutil.move(mesh, mesh_dir)
            generated = True
        """
    bar.next()
bit = df["bitmap"]
plt.imshow(bit.mean(), cmap='gray')

curve = df["area"]
print(df["bitmap"])

# End bar
bar.finish()

# %%
