# Generic imports
#%%
import os
import time

import shutil
from   datetime import datetime
from scipy.stats import qmc
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
import shapes
import meshes
import importlib
import multiprocessing
import time
from multiprocessing import Pool


importlib.reload(shapes)
importlib.reload(meshes)

from shapes import *
from meshes import *

### ************************************************
### Generate full dataset
# Parameters
from progress.bar import Bar
import progress.bar

# Generic imports
#%%
import os
import random
import shutil
import progress.bar
from   datetime import datetime
from scipy.stats import qmc
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft

import shapes
import meshes
import importlib

importlib.reload(shapes)
importlib.reload(meshes)

from shapes import *
from meshes import *

### ************************************************
### Generate full dataset
# Parameters
def generate_data_indicators_light_ind(seeds,number_of_samples):
    bit, df1 = generate_samples()
    points = df1["curve_points"]
    df1["angles"] = None
    df1["angles"] = df1["angles"].astype(object)
    df1["spectra"] = None
    df1["spectra"] = df1["spectra"].astype(object)
    limit = 20
    for index,row in df1.iterrows():
        points = row["curve_points"]
        angles = np.arctan2(points[:, 1], points[:, 0])
        df1.at[index, "angles"] = angles
        og_data = np.linalg.norm(points, axis = 1)

        fs1 = angles.shape[0]/(2*np.pi)
        samples1 = og_data
        fft1 = fft(og_data)
        freqs1 = fftfreq(len(samples1), 1/fs1)
        df1.at[index,"spectra"] = fft1[:limit]/points.shape[0]

    spectra = df1["spectra"]
    spectra = np.stack(spectra.values)
    
    return df1,spectra

def generate_samples(numero_samples,n_base,code, latin_gen_bool = False, bit_im = False, bitmap_generate = False):
    n_sampling_pts = 20
    mesh_domain    = False
    plot_pts       = True
    show_quadrants = True
    n_shapes       = numero_samples
    time           = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    dataset_dir    = 'dataset_'+time+'/'
    mesh_dir       = dataset_dir+f'meshes_{code}/'
    img_dir        = dataset_dir+f'images_{code}/'
    det_img_dir    = dataset_dir+f'det_images_{code}/'
    bit_img_dir    = dataset_dir+f'bit_images_{code}/'
    save_det_plot = False

    filename       = 'shape'
    magnify        = 1.0
    max_radius = (2)**0.5
    xmin           =-2.0
    xmax           = 2.0
    ymin           =-2.0
    ymax           = 2.0
    n_tri_max      = 5000
    n_pts = n_base

    ### generate latin hypercube points in the defined space
    equ_dim_lim = 1.5

    if latin_gen_bool != False:
        latin_gen = qmc.LatinHypercube(d=n_pts*2)
        latin_gen = latin_gen.random(n=n_shapes)
        latin_gen = equ_dim_lim * latin_gen

        latin_gend2 = qmc.LatinHypercube(d=n_pts*2)
        latin_gend2 = latin_gend2.random(n=n_shapes)
        latin_gend2 = latin_gend2 * 0.9
        radius_tot = latin_gend2.reshape(-1,n_shapes,n_pts)
        edgy_tot   = latin_gend2.reshape(-1,n_shapes,n_pts)
    else:
        latin_gen = False


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
            if not latin_gen_bool:
                #n_pts  = random.randint(3, 7)
                radius = np.random.uniform(0.1, .3, size=n_pts)
                print(radius)
                edgy   = np.random.uniform(0.1, .3, size=n_pts)
            else:
                radius = radius_tot[0,i,:]
                edgy   = radius_tot[1,i,:]

            shape  = Shape(filename+'_'+str(i),
                        None,
                        n_pts,
                        n_sampling_pts,
                        radius,
                        edgy,
                        save_det_plot = save_det_plot)
            
            shape.generate(magnify=1.0,
                        xmin=xmin,
                        xmax=xmax,
                        ymin=ymin,
                        ymax=ymax,
                        latin_gen = latin_gen)
            
            if save_det_plot:
                img  = filename+'_'+str(i)+"det_plot"+'.png'
                shutil.move(img,  det_img_dir)

            if not bitmap_generate:
                shape.bitmap = 0

            else:
                shape.generate_bitmap(bit_im=bit_im)
            
            shape.generate_bitmap(bit_im=bit_im)

            if bit_im:
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
    curve = df["area"]
    # End bar
    bar.finish()
    return bit,df

class analysis:
    def __init__ (self, n_shapes = 10,n_base = 10 ,latin_gen_bool = False):
        self.pre = ""
        self.n_sampling_pts = 30
        self.mesh_domain    = False
        self.plot_pts       = True
        self.show_quadrants = True
        self.n_shapes       = n_shapes
        self.n_base = n_base
        self.time           = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        self.dataset_dir    = 'dataset_' + self.time + '/'
        self.mesh_dir       = self.dataset_dir + 'meshes/'
        self.img_dir        = self.dataset_dir + 'images/'
        self.det_img_dir    = self.dataset_dir + 'det_images/'
        self.bit_img_dir    = self.dataset_dir + 'bit_images/'
        self.save_det_plot  = False
        self.latin_gen_bool = latin_gen_bool

        self.filename       = 'shape'
        self.magnify        = 1.0
        self.max_radius     = (2)**0.5
        self.xmin           = -2.0
        self.xmax           =  2.0
        self.ymin           = -2.0
        self.ymax           =  2.0
        self.n_tri_max      = 5000
        self.equ_dim_lim    = 1.5
        self.current_indicators = 0

    def resetdirectory(self):
        self.time           = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        self.dataset_dir    = f'{self.pre}dataset_' + self.time + '/'
        self.mesh_dir       = self.dataset_dir + 'meshes/'
        self.img_dir        = self.dataset_dir + 'images/'
        self.det_img_dir    = self.dataset_dir + 'det_images/'
        self.bit_img_dir    = self.dataset_dir + 'bit_images/'

    def generate_samples(self, bit_im = False, bitmap_generate = False):
        self.resetdirectory()
        latin_gen = None

        if self.latin_gen_bool:
            latin_gen = qmc.LatinHypercube(d=self.n_base * 2)
            latin_gen = latin_gen.random(n=self.n_shapes)
            latin_gen = self.equ_dim_lim * latin_gen

            radius_tot = latin_gen.reshape(-1, self.n_shapes, self.n_base)
            edgy_tot   = latin_gen.reshape(-1, self.n_shapes, self.n_base)
            
        
        if not os.path.exists(self.mesh_dir):
            os.makedirs(self.mesh_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        if not os.path.exists(self.det_img_dir):
            os.makedirs(self.det_img_dir)
        if not os.path.exists(self.bit_img_dir):
            os.makedirs(self.bit_img_dir)

        bar = progress.bar.Bar('Generating shapes', max=self.n_shapes)
        df = pd.DataFrame(columns=['curve_points', 'area', 'perimeter','bitmap'])

        for i in range(0, self.n_shapes):
            generated = False
            while not generated:
                if not self.latin_gen_bool:
                    radius = np.random.uniform(0.7, .9, size=self.n_base)
                    edgy   = np.random.uniform(0.1, .3, size=self.n_base)
                    print(radius)
                else:
                    radius = radius_tot[0, i, :]
                    edgy   = edgy_tot[1, i, :]

                shape = Shape(self.filename + '_' + str(i),
                            None,
                            self.n_base,
                            self.n_sampling_pts,
                            radius,
                            edgy,
                            save_det_plot=self.save_det_plot)
                
                shape.generate(magnify=self.magnify,
                            xmin=self.xmin,
                            xmax=self.xmax,
                            ymin=self.ymin,
                            ymax=self.ymax,
                            latin_gen=self.latin_gen_bool,
                            latin_gen_outer_space=latin_gen)
                
                if self.save_det_plot:
                    img = self.filename + '_' + str(i) + "det_plot" + '.png'
                    shutil.move(img, self.det_img_dir)

                if not bitmap_generate:
                    shape.bitmap = 0

                else:
                    shape.generate_bitmap(bit_im=bit_im)


                if bit_im:
                    img = self.filename + '_' + str(i) + "bit" + '.png'
                    shutil.move(img, self.bit_img_dir)

                new_rows = [{
                    'curve_points': shape.curve_pts,
                    'area': shape.area,
                    'perimeter': shape.curve_length,
                    'bitmap': shape.bitmap
                }]
                df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
                generated = True

            bar.next()

        bit = df["bitmap"]
        curve = df["area"]
        bar.finish()
        return bit, df

    def generate_data_indicators(self):
        list_mse_mean = np.array([])
        list_mse_var = np.array([])
        bit, df1 = self.generate_samples()
        stack1 = np.stack(bit.values)
        mean1_img = np.mean(stack1, axis=0)
        mse1 = np.mean((stack1 - mean1_img) ** 2, axis=(1, 2))
        mean1 = np.mean(mse1)
        var1 = np.var(mse1)
        list_mse_mean = np.append(list_mse_mean, mean1)
        list_mse_var = np.append(list_mse_var, var1)

        points = df1["curve_points"]
        df1["angles"] = None
        df1["angles"] = df1["angles"].astype(object)
        df1["spectra"] = None
        df1["spectra"] = df1["spectra"].astype(object)
        limit = 20
        for index,row in df1.iterrows():
            points = row["curve_points"]
            angles = np.arctan2(points[:, 1], points[:, 0])
            df1.at[index, "angles"] = angles
            og_data = np.linalg.norm(points, axis = 1)

            fs1 = angles.shape[0]/(2*np.pi)
            samples1 = og_data
            fft1 = fft(og_data)
            freqs1 = fftfreq(len(samples1), 1/fs1)
            df1.at[index,"spectra"] = fft1[:limit]/points.shape[0]

        spectra = df1["spectra"]
        spectra = np.stack(spectra.values)
        var_energy = spectra.var(axis = 0)
        mean_energy = abs(spectra).mean(axis = 0)

        var_angle = np.angle(spectra).var(axis = 0)
        mean_angle = np.angle(spectra).mean(axis = 0)

        self.current_indicators = (var_energy, mean_energy, var_angle, mean_angle, df1,spectra)
        return var_energy, mean_energy, var_angle, mean_angle, df1,spectra

    def generate_data_indicators_light(self):
        bit, df1 = self.generate_samples()
        points = df1["curve_points"]
        df1["angles"] = None
        df1["angles"] = df1["angles"].astype(object)
        df1["spectra"] = None
        df1["spectra"] = df1["spectra"].astype(object)
        limit = 20
        for index,row in df1.iterrows():
            points = row["curve_points"]
            angles = np.arctan2(points[:, 1], points[:, 0])
            df1.at[index, "angles"] = angles
            og_data = np.linalg.norm(points, axis = 1)

            fs1 = angles.shape[0]/(2*np.pi)
            samples1 = og_data
            fft1 = fft(og_data)
            freqs1 = fftfreq(len(samples1), 1/fs1)
            df1.at[index,"spectra"] = fft1[:limit]/points.shape[0]

        spectra = df1["spectra"]
        spectra = np.stack(spectra.values)
        
        return df1,spectra

    def run_study_var_score(self, number_of_sampes = 50,seed_interval = [3,7]):
        score_array = []
        df1_arrays = []
        plt.figure()  # rOnly call this once
        list_of_seeds = np.arange(seed_interval[0], seed_interval[1])
        for seed_points in list_of_seeds:
            var_energy_array = []
            mean_energy_array = []
            var_angle_array = []
            mean_angle_array = []
            self.n_base = seed_points
            for elements in range(number_of_sampes):
                var_energy, mean_energy, var_angle, mean_angle,df1, spectra = self.generate_data_indicators()
                var_energy_array.append(var_energy[1:])
                mean_energy_array.append(mean_energy[1:])
                var_angle_array.append(var_angle)
                mean_angle_array.append(mean_angle)
                time.sleep(0.01)  # delays for 2 seconds
                df1_arrays.append(df1)

            total_score = np.array(var_energy_array)**0.5 * np.array(mean_energy_array)
            total_score = total_score.sum(axis = 1)
            score_array.append(total_score)

        return score_array, df1_arrays

    def run_study_mindiff_score(self, number_of_samples = 50, seed_interval = [3,7]):
        self.pre = f"min_dataset{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}/"

        score_array = []
        df1_arrays = []
        plt.figure()  # rOnly call this once
        list_of_seeds = np.arange(seed_interval[0], seed_interval[1])
        bar = Bar('Processing', max=seed_interval[1] * number_of_samples)
        self.n_shapes = number_of_samples
        for seed_points in list_of_seeds:
            self.n_base = seed_points
            mean_diff_array = []
            for elements in range(number_of_samples):
                var_energy, mean_energy, var_angle, mean_angle,df1, spectra = self.generate_data_indicators()
                df1_arrays.append(df1)

                arr = spectra
                min_dif = self.find_min_dataset(arr)
                time.sleep(0.01)
                bar.next()

                mean_diff = min_dif.mean()
                mean_diff_array.append(mean_diff)
            score_array.append(mean_diff_array)
        bar.finish()
        self.pre = ""

        return score_array, df1_arrays
    

    def find_min_dataset(self,spectra):
        arr = spectra
        min_dif = np.zeros(arr.shape[0])
        for elements in range(arr.shape[0]):
            diff = np.linalg.norm(arr - arr[elements],axis = 1)
            min_dif[elements] = np.min(diff[diff != 0])
        return min_dif

    def spectrogram(self, seed_interval = [3,7]):
        list_of_seeds = np.arange(seed_interval[0], seed_interval[1])
        array_of_indicators  = []
        for seed_points in list_of_seeds:
            self.n_base = seed_points
            tups_indicator = self.generate_data_indicators()
            array_of_indicators.append(tups_indicator)
        return array_of_indicators

    def uniform_generation(self, number_of_samples = 1000, seed_points = [3,10]):
        samples_per_gen = number_of_samples/(seed_points[1] - seed_points[0])
        i = 0
        spectra_total = 0
        df_total = 0
        self.n_shapes = int(samples_per_gen)
        for elements in np.arange(seed_points[0], seed_points[1]):
            if elements == (seed_points[1] - 1):
                self.n_shapes += number_of_samples%(seed_points[1] - seed_points[0])
            self.n_base = elements
            self.n_sampling_pts
            df1, spectra = self.generate_data_indicators_light()

            if i == 0 :
                i = 1
                spectra_total = spectra
                df_total = df1

            else:
                spectra_total = np.concatenate([spectra_total, spectra])
                df_total = pd.concat([df_total, df1], ignore_index=True)

        return spectra_total, df_total

    def weighted_generation(self, number_of_samples = 1000, seed_points = [3,10], weight = np.zeros(7) + (1/7)):
        samples_per_gen = np.array(weight * number_of_samples,dtype = int)
        residual = 1000 - samples_per_gen.sum()
        i = 0
        spectra_total = 0
        df_total = 0
        for elements in np.arange(seed_points[0], seed_points[1]):
            self.n_shapes = samples_per_gen[i]
            if elements == (seed_points[1] - 1):
                self.n_shapes += residual
            self.n_base = elements
            df1, spectra = self.generate_data_indicators_light()

            if i == 0 :
                spectra_total = spectra
                df_total = df1
            else:
                spectra_total = np.concatenate([spectra_total, spectra])
                #df_total = pd.concat([df_total, df1], ignore_index=True)
            i += 1

        return spectra_total, df_total
        
    def weighted_generation_parallel(self, number_of_samples = 1000, seed_points = [3,10], weight = np.zeros(7) + (1/7)):
        samples_per_gen = np.array(weight * number_of_samples,dtype = int)
        residual = number_of_samples - samples_per_gen.sum()
        samples_per_gen[-1] = samples_per_gen[-1] + residual
        seeds = np.arange(seed_points[0], seed_points[1])
        spectra_total = 0
        args = list(zip(samples_per_gen, seeds,seeds))
        with Pool() as pool:
            results = pool.starmap(generate_data_indicators_light_ind, args)
        return results
    
def generate_data_indicators_light_ind(seeds,number_of_samples,code):
    bit, df1 = generate_samples(seeds,number_of_samples,code)
    points = df1["curve_points"]
    df1["angles"] = None
    df1["angles"] = df1["angles"].astype(object)
    df1["spectra"] = None
    df1["spectra"] = df1["spectra"].astype(object)
    limit = 20
    for index,row in df1.iterrows():
        points = row["curve_points"]
        angles = np.arctan2(points[:, 1], points[:, 0])
        df1.at[index, "angles"] = angles
        og_data = np.linalg.norm(points, axis = 1)

        fs1 = angles.shape[0]/(2*np.pi)
        samples1 = og_data
        fft1 = fft(og_data)
        freqs1 = fftfreq(len(samples1), 1/fs1)
        df1.at[index,"spectra"] = fft1[:limit]/points.shape[0]

    spectra = df1["spectra"]
    spectra = np.stack(spectra.values)
    
    return spectra

# %%
