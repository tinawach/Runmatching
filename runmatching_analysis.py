import sys
import numpy as np
import matplotlib as mpl
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
from regions import CircleSkyRegion, RectangleSkyRegion
from gammapy.maps import WcsGeom, MapAxis, Map, WcsNDMap
from gammapy.makers import MapDatasetMaker, SafeMaskMaker, FoVBackgroundMaker
from gammapy.data import DataStore
from gammapy.datasets import Datasets, FluxPointsDataset, MapDataset
from gammapy.modeling import Fit
from gammapy.modeling.models import SkyModel, LogParabolaSpectralModel, PointSpatialModel, PowerLawSpectralModel
from gammapy.estimators import FluxPoints,FluxPointsEstimator, ExcessMapEstimator
from gammapy.modeling.models import FoVBackgroundModel

import sys  
sys.path.insert(1, '/home/hpc/caph/mppi103h/Documents/On-OFF-matching-woody')
import common_utils
from common_utils import get_excluded_regions

#load the auxiliary information of the runs
DB_general = pd.read_csv('/home/wecapstor1/caph/shared/hess/fits/database_image/data_1223.csv', header=0)

# load the zenith correction factors
correction_factor = pd.read_csv('/home/wecapstor1/caph/mppi103h/On-Off-matching/zenith_correction_factors.csv', sep='\t')

#specific optical phases for the background template of the HESS telescopes
muon_phases2 = [20000, 40093, 57532, 60838, 63567, 68545, 80000, 85000, 95003,
                100800, 110340, 127700, 128600, 132350, 154814, 180000]




class matching_dataset():
    """
    Compute the dataset for the run matching analysis
    -----------------------------------------------------------------------------------------
    Parameters:

        - ds: gammapy datasetore object, containing all relevant observation fits files
    
        - obs_list: list, observation Ids of the observations that should be stacked into the dataset
    
        - obs_list_off: list, observation Ids of the OFF observations from which the background is estimated
    
        - geom: gammapy.MapDataSet.geom Object, containing the geometry of the analysis region
    
        - energy_axis_true: MapAxis object, true energy axis of the geometry used for the analysis
    
        - offset_max: float, maximum event offset from the center of the camera

    """


    def __init__(self, systematic_shift, ds, obs_list, obs_list_off, deviation, geom, energy_axis_true, offset_max):
        
        self.ds = ds
        self.obs_list = obs_list
        self.obs_list_off = obs_list_off
        self.deviation = deviation
        self.geom = geom
        self.energy_axis_true = energy_axis_true
        self.offset_max = offset_max
        self.systematic_shift = systematic_shift
        

    

    def livetime_corr(self, obs, off_run, dataset):
        """ Correct for the differences in deadtime corrected observation time """
        
        livetime_dev = off_run.observation_live_time_duration.value - obs.observation_live_time_duration.value
        counts_per_sec = dataset.background.data/off_run.observation_live_time_duration.value
        factors = counts_per_sec*livetime_dev
        bkg = [x - y for x, y in zip(dataset.background.data, factors)]

        bkg = WcsNDMap(geom=dataset.counts.geom, data=np.array(bkg))
        dataset.background = bkg
        return dataset
    
    
    def zenith_corr(self, obs, off_run, dataset):
        """ Correct for the differences in zenith angle """
        
        for phase in range(0,len(muon_phases2)-1):
            if obs.obs_id > muon_phases2[phase] and obs.obs_id < muon_phases2[phase+1]:
                factor = correction_factor.x_2.iloc[phase]
    
        bkg = dataset.background.data
        zenith_off = np.deg2rad(self.ds.obs_table[self.ds.obs_table['OBS_ID']==off_run.obs_id]["ZEN_PNT"])
        zenith_on = np.deg2rad(self.ds.obs_table[self.ds.obs_table['OBS_ID']==obs.obs_id]["ZEN_PNT"])
        bkg = np.array(bkg) * np.cos(zenith_on - zenith_off)**factor
        
        bkg = WcsNDMap(geom=dataset.counts.geom, data=np.array(bkg))
        dataset.background = bkg
        return dataset

    
    def systematics(self, obs, off_run, dataset, sys):
        """ Get the upper or lower limit of the background rate from the systematic error on the background template and run matching """
        run_dev = self.deviation[self.obs_list.index(obs)]
        bkg = dataset.background.data
        binsz = (self.systematic_shift['run dev'].iloc[1] - self.systematic_shift['run dev'].iloc[0])/2
        for i in range(0,len(self.systematic_shift)):
            if run_dev > self.systematic_shift['run dev'].iloc[i]-binsz and run_dev < self.systematic_shift['run dev'].iloc[i]+binsz:
                index = i
                sys_factor = self.systematic_shift['sys dev error'].iloc[i]
                if sys_factor == 0:
                    closest_filled = index - min((index - self.systematic_shift[self.systematic_shift['sys dev'] != 0].index), key=abs)
                    sys_factor = self.systematic_shift['sys dev error'].iloc[closest_filled]
            
            elif run_dev < self.systematic_shift['run dev'].iloc[0]:
                sys_factor = self.systematic_shift['sys dev error'].iloc[0]

            elif run_dev > self.systematic_shift['run dev'].iloc[-1]:
                sys_factor = self.systematic_shift['sys dev error'].iloc[-1]
        
        if sys == 'low':
            bkg = [x - x*sys_factor for x in bkg] 
        elif sys == 'high':
            bkg = [x + x*sys_factor for x in bkg]   
        
        bkg = WcsNDMap(geom=dataset.counts.geom, data=np.array(bkg))
        dataset.background = bkg
        return dataset
        

    def find_energy_threshold(self, dataset):
        number = 0
        for k in range(0,24): 
            for l in range(0,200):
                for n in range(0,200):
                    if dataset.mask_safe.data[k][l][n] == True:
                        number = k 
                        break
            if number != 0:
                break
        return number

    def get_exclusion_mask(self, off_run, geom_off, radius):
        """ Define the exclusion mask that will be used for the background fit """ 
        
        hap_exclusion_regions = get_excluded_regions(geom_off.center_coord[0].value, geom_off.center_coord[1].value, radius)
        excl_regions = []
        for source in hap_exclusion_regions:
            center = SkyCoord(source.ra, source.dec, unit='deg', frame='icrs')
            region = CircleSkyRegion(center=center, radius=source.radius*u.deg)
            excl_regions.append(region)

        tel_pointing = off_run.pointing.fixed_icrs
        source_pos = DB_general[DB_general['Run']==off_run.obs_id]
        ra = tel_pointing.ra.value - source_pos['Offset_x'].iloc[0]
        dec = tel_pointing.dec.value - source_pos['Offset_y'].iloc[0]
        
        excl_regions.append(CircleSkyRegion(center=SkyCoord(ra*u.deg, dec*u.deg), radius=0.3*u.deg))
        data2 = geom_off.region_mask(regions=excl_regions, inside=False)
        maker_fov_off = FoVBackgroundMaker(method="fit", exclusion_mask=data2)
        ex = maker_fov_off.exclusion_mask.cutout(off_run.pointing.fixed_icrs,
                                                 width=2 * self.offset_max)
        return maker_fov_off, ex

    
    def adjust_energy_threshold(self, dataset, number):
        """ Apply a predifined save energy to the counts and background of a dataset """
        
        bkg_array = np.zeros_like(dataset.data)
        for a in range(number, bkg_array.shape[0]):
            for b in range(0,200):
                for c in range(0,200):
                    bkg_array[a][b][c] = dataset.data[a][b][c]
        
        masked_data = WcsNDMap(geom=dataset.geom, data=np.array(bkg_array))
        return masked_data

    def compute_empty_dataset(self):
        
        stacked = MapDataset.create(
        geom = self.geom, 
        name = "Crab",  
        energy_axis_true = self.energy_axis_true
        )
        
        return stacked
    
    
    def compute_matched_dataset(self, corrections=None, systematics=None, debug=0):
        """
        Compute the background rate expected for our ON region from an OFF observation and create the dataset
    
        -----------------------------------------------------------------------------------------
        Parameters:
    
        - corrections: string, choose between 'none', 'livetime', 'zenith', 'all' 
    
        - systematics: string, choose between 'low' and 'high' 

        - debug: float, choose between '0' (no debugging) or '1' (returns a list of the mean of the significance distribution for each run)
    
        """

        sig_dist = []
        stacked_dataset = self.compute_empty_dataset()

        maker = MapDatasetMaker()
        maker_fov = FoVBackgroundMaker(method="fit")
        maker_safe_mask2 = SafeMaskMaker(
            methods=["offset-max", 'aeff-default', 'aeff-max', 'edisp-bias', 'bkg-peak'],
            offset_max=self.offset_max,
            bias_percent=10
        )
        
    
        for m in range(0, len(self.obs_list)):
            obs = self.obs_list[m]
            off_run = self.obs_list_off[m]
            
            # ON run dataset geometry 
            cutout = stacked_dataset.cutout(
                obs.pointing.fixed_icrs, 
                width=2 * self.offset_max, 
                name=f"obs-{obs.obs_id}"
            )
            dataset = maker.run(cutout, obs)
            dataset = maker_safe_mask2.run(dataset, obs)   
        
            number_on = self.find_energy_threshold(dataset)

            
            # OFF run dataset geometry:
            geom_off = WcsGeom.create(skydir=off_run.pointing.fixed_icrs,
                                      binsz=0.02,
                                      width=(4.0, 4.0),
                                      frame="icrs", proj="CAR",
                                      axes=[self.geom.axes['energy']]
                                     )
            
            cutout_off = MapDataset.create(geom=geom_off,
                                           name=f"obs-{obs.obs_id}",
                                           energy_axis_true= self.energy_axis_true
                                          )

            
            #define exclusion region and safe mask for the background fit
            maker_fov_off, ex = self.get_exclusion_mask(off_run, geom_off, 5)

            
            #fit the background to the OFF run
            dataset_off = maker.run(cutout_off, off_run)
            dataset_off = maker_safe_mask2.run(dataset_off, off_run)
            
            number_off = self.find_energy_threshold(dataset_off)

            #find the highest energy threshold between ON and OFF run
            number = max(number_on, number_off)

            #apply the energy threshold to both ON and OFF run
            dataset.background = self.adjust_energy_threshold(dataset.background, number)
            dataset.counts = self.adjust_energy_threshold(dataset.counts, number)
            dataset_off.background = self.adjust_energy_threshold(dataset_off.background, number)
            dataset_off.counts = self.adjust_energy_threshold(dataset_off.counts, number)


            #The background fit
            bkg_model = FoVBackgroundModel(dataset_name=dataset_off.name)
            dataset_off.models=bkg_model
            dataset_off.background_model.spectral_model.tilt.frozen = False
            dataset_off = maker_fov_off.run(dataset_off)
        
    
            bkg_model = FoVBackgroundModel(dataset_name=dataset.name)
            dataset.models=bkg_model
        
            dataset.background_model.spectral_model.norm.value = dataset_off.background_model.spectral_model.norm.value
            dataset.background_model.spectral_model.tilt.value = dataset_off.background_model.spectral_model.tilt.value
            dataset.background_model.spectral_model.norm.frozen = True
            dataset.background_model.spectral_model.tilt.frozen = True
        
            
            #Apply the corrections
            
            if corrections == 'livetime':
                dataset = self.livetime_corr(obs, off_run, dataset)
            elif corrections == 'zenith':
                dataset = self.zenith_corr(obs, off_run, dataset)
            elif corrections == 'all':
                dataset = self.livetime_corr(obs, off_run, dataset)
                dataset = self.zenith_corr(obs, off_run, dataset)
            
            if systematics == 'low':
                dataset = self.systematics(obs, off_run, dataset, sys='low')
            elif systematics == 'high':
                dataset = self.systematics(obs, off_run, dataset, sys='high')

            
            print(obs.obs_id)
            #print(off_run.obs_id)

            if debug==1:
                #Significance of the model:
                estimator_001 = ExcessMapEstimator(
                correlation_radius="0.1 deg",
                # energy_edges=[0.3, 100] * u.TeV,
                selection_optional=[],)
                lima_maps_001 = estimator_001.run(dataset)
          
                bins=np.linspace(-6, 8, 131)
                significance_map_off = lima_maps_001["sqrt_ts"]
                significance_off = significance_map_off.data[np.isfinite(significance_map_off.data)]
                significance_all = significance_map_off.data[np.isfinite(significance_map_off.data)]
                significance_data = significance_off
                mu, std = norm.fit(significance_data)
                sig_dist.append([obs.obs_id, off_run.obs_id, mu])
        
            
            stacked_dataset.stack(dataset)

            if debug==2: 
                stacked_dataset.counts.get_spectrum().plot()
                stacked_dataset.background.get_spectrum().plot()
                plt.show()
        
        if debug == 0 or debug==2:
            return stacked_dataset
        if debug == 1:
            return stacked_dataset, sig_dist




    def standard_dataset(self, debug=0):
        """
        Compute a dataset using the 3-dimensional background template directly on the target observations
        -----------------------------------------------------------------------------------------
        Parameters:
    
            - debug: float, choose between '0' (no debugging) or '1' (returns a list of the mean of the significance distribution for each run)
    
        """

        stacked_dataset = self.compute_empty_dataset()

        sig_dist = []
        maker = MapDatasetMaker()
        maker_safe_mask = SafeMaskMaker(
            methods=["offset-max", 'aeff-default', 'aeff-max', 'edisp-bias', "bkg-peak"],
            offset_max= self.offset_max,
            bias_percent=10
        )
        
        for obs in self.obs_list:
            
            cutout = stacked_dataset.cutout(
                obs.pointing.fixed_icrs,
                width=2 * self.offset_max,
                name=f"obs-{obs.obs_id}"
            )
            
            maker_fov, ex = self.get_exclusion_mask(obs, self.geom, 5)
    
            
            dataset = maker.run(cutout, obs)
            dataset = maker_safe_mask.run(dataset, obs)   
            
            # fit background model
            bkg_model = FoVBackgroundModel(dataset_name=dataset.name)
            dataset.models=bkg_model
            dataset.background_model.spectral_model.tilt.frozen = False
            dataset = maker_fov.run(dataset)
            
            if debug == 1:
                estimator_001 = ExcessMapEstimator(
                correlation_radius="0.1 deg",
                # energy_edges=[0.3, 50] * u.TeV,
                selection_optional=[],
                )
                lima_maps_001 = estimator_001.run(dataset)
               
                significance_map_off = lima_maps_001["sqrt_ts"]
                significance_off = significance_map_off.data[np.isfinite(significance_map_off.data)]
                significance_all = significance_map_off.data[np.isfinite(significance_map_off.data)]
                significance_data = significance_off
                mu, std = norm.fit(significance_data)
                sig_dist.append([obs.obs_id, mu])
            
            stacked_dataset.stack(dataset)

        if debug == 0:
                return stacked_dataset
        if debug == 1:
            return stacked_dataset, sig_dist




class significance_visualization():

    """ 
     Create a visual comparison of the region before and after the model fitting for both background estimation techniques 
    -----------------------------------------------------------------------------------------
        Parameters:
    
            - sigmap_pre: WcsNDMap, significance map before the model fitting

            - sigmap_post: WcsNDMap, significance map after model fitting

            - cmap: string, colormap used for plotting

            - levels: list, levels of the significance contours

            - color: string, color to visualize the significance contours

            - model: the source model you want to overplot on the post fit significance map

            - modelcolor: color for plotting the best-fit model contours
    
    """
    
    def __init__(self, sigmap_pre, sigmap_post, cmap, levels, color, model =None, modelcolor=None):
        self.sigmap_pre = sigmap_pre
        self.sigmap_post = sigmap_post
        self.cmap = cmap
        self.levels = levels
        self.color = color
        self.model = model
        self.modelcolor = modelcolor


    def plot_model(self, ax2):

        if self.model.names[0] != 'template':

            modeltype = self.model.to_dict()['components'][0]['spatial']['type']
            
            if modeltype == 'PointSpatialModel':
                geom = self.sigmap_post.geom
                lon = self.model.to_dict()['components'][0]['spatial']['parameters'][0]['value']
                lat = self.model.to_dict()['components'][0]['spatial']['parameters'][1]['value']
                center = SkyCoord(lon * u.deg, lat * u.deg)
                
                pix_coords = skycoord_to_pixel(center, geom.wcs)
                ax2.scatter(pix_coords[0], pix_coords[1], s=100, c=self.modelcolor, marker='x')
        
            if modeltype == 'GaussianSpatialModel' or modeltype == 'DiskSpatialModel':
                geom = self.sigmap_post.geom
                lon = self.model.to_dict()['components'][0]['spatial']['parameters'][0]['value']
                lat = self.model.to_dict()['components'][0]['spatial']['parameters'][1]['value']
                radius = self.model.to_dict()['components'][0]['spatial']['parameters'][2]['value']
                center = SkyCoord(lon * u.deg, lat * u.deg)
        
                circle = CircleSkyRegion(center, radius = radius*u.deg)
                source_fit = circle.to_pixel(geom.to_image().wcs)
                
                source_fit.plot(linestyle='--', linewidth=2, edgecolor=self.modelcolor, facecolor='None')

        else:
            pass


    def significance_map_comparison(self, levels):
        """ 
         Create a side by side visualization 
        -----------------------------------------------------------------------------------------
            Parameters:
        
                - levels: list, contains the contour levels which are displayed
        
        """
        
        fig = plt.figure(figsize=(10, 10))
    
        ax1 = plt.subplot(221, projection = self.sigmap_pre.geom.wcs)
        ax2 = plt.subplot(222, projection = self.sigmap_post.geom.wcs)
    
        ax1.set_title("Significance map before the fit")
        self.sigmap_pre.plot(ax=ax1, add_cbar=True, cmap=self.cmap)
        CS = ax1.contour(np.squeeze(self.sigmap_pre.sum_over_axes().data, axis=(0,)),
                         levels, colors=self.color, alpha=1, linewidths=1.3)
        
        ax2.set_title("Significance map after the fit")
        self.sigmap_post.plot(ax=ax2, add_cbar=True, cmap=self.cmap)
        CS = ax2.contour(np.squeeze(self.sigmap_post.sum_over_axes().data, axis=(0,)),
                         levels, colors=self.color, alpha=1, linewidths=1.3)
        
        self.plot_model(ax2)


    def significance_contours_comparison(self, sigmap_pre2, levels, color1, color2):
        """ 
         Directly compare the contours of both bkg estimation techniques in one plot
        -----------------------------------------------------------------------------------------
            Parameters:

                - sigmap2_pre: gammapy WCSND Map, containing the information for the computation of the second set of contours
        
                - levels: list, contains the contour levels which are displayed

                - color1: string, color in which the RM dataset contours are displayed

                - color2: string, color in which the FOV dataset contours are displayed
        
        """
    
    
        fig = plt.figure(dpi=150)
        
        significance_RM = np.squeeze(self.sigmap_pre.sum_over_axes().data, axis=(0,))
        significance_FOV = np.squeeze(sigmap_pre2.sum_over_axes().data, axis=(0,))
        
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], 
                          projection=self.sigmap_pre.geom.wcs)
        
        plt.rcParams.update({'font.size': 10})
        plt.rcParams["font.weight"] = "normal"
        plt.rcParams["axes.labelweight"] = "normal"
        
        geom= self.sigmap_pre.geom
        plot = ax.imshow(significance_RM, cmap=self.cmap, origin='lower')
        
        cbar = fig.colorbar(plot, ax=ax)
        cbar.ax.set_ylabel(r'significance [$\sigma$]', rotation=90, fontsize=11)
        cbar.ax.tick_params(labelsize=10)
                
        
        CS = ax.contour(significance_RM, levels, colors=color1, alpha=1, linewidths=1.3)
        CS = ax.contour(significance_FOV, levels, colors=color2, alpha=1, linestyles='--', linewidths=1.3)
        
        
        self.plot_model(ax)
        
        
        ax.set_ylabel('Declination')
        ax.set_xlabel('Right Ascension')


    def distribution_comparison(self, sigmap_pre2, color1, color2, name_dist1, name_dist2):

        bins=np.linspace(-6, 11, 151)

        hap_exclusion_regions = get_excluded_regions(sigmap_pre2.geom.center_coord[0].value, sigmap_pre2.geom.center_coord[1].value, 5)
        excl_regions = []
        for source in hap_exclusion_regions:
            center = SkyCoord(source.ra, source.dec, unit='deg', frame='icrs')
            region = CircleSkyRegion(center=center, radius=source.radius*u.deg)
            excl_regions.append(region)
                   
        excl_data = self.sigmap_pre.geom.region_mask(regions=excl_regions, inside=False)
        significance_data = self.sigmap_pre * excl_data
        significance_data = significance_data.data[np.isfinite(self.sigmap_pre.data)]
        significance_data = [x for x in significance_data if x!=0]
        
        excl_data = sigmap_pre2.geom.region_mask(regions=excl_regions, inside=False)
        significance_data2 = sigmap_pre2 * excl_data
        significance_data2 = significance_data2.data[np.isfinite(sigmap_pre2.data)]
        significance_data2 = [x for x in significance_data2 if x!=0]
        
        
        #fit data: 
        n2, bins2, patches2 = plt.hist(significance_data2, bins=bins, histtype='step', 
                                                color='k', lw=2, zorder=1)
        
        n, bins, patches = plt.hist(significance_data, bins=bins, histtype='step', 
                                                color='dimgray', lw=2, zorder=1)
        
        
        x_vals = 0.5*(bins[:-1]+bins[1:])
        mu, std = norm.fit(significance_data)
        mu_opt, std_opt = norm.fit(significance_data2)
        p = norm.pdf(x_vals, mu, std)
        p_opt = norm.pdf(x_vals, mu_opt, std_opt)
        
        
        #plot:
        y = norm.pdf(x_vals, mu, std) * sum(n * np.diff(bins))
        y2 = norm.pdf(x_vals, mu_opt, std_opt) * sum(n * np.diff(bins))
        
        plt.plot(x_vals, y2, lw=2, color=color1,
                 label=r"{}: $\mu$ = {:.2f}, $\sigma$ = {:.2f}".format(name_dist1, mu_opt, std_opt),
                 zorder=2)
        plt.plot(x_vals, y, lw=2,color=color2,
                 label=r"{}: $\mu$ = {:.2f}, $\sigma$ = {:.2f}".format(name_dist2, mu, std),)
        
        plt.grid(ls='--')
        plt.legend(loc=1, fancybox=True, shadow=True)
        plt.xlim(-5, 5.6)
        plt.ylim(10**0, 4*10**4)
        plt.ylabel('Counts')
        plt.xlabel('Significance')
        plt.yscale('log')


    
    
def read_fluxpoints_from_csv(fluxpoints):
    """ Convert flux from a csv file into energy flux and return the SED as a pd.DataFrame """

    columns = ['e', 'e_min', 'e_max', 'flux', 'flux_min', 'flux_max', 'flag']
    df = pd.DataFrame(fluxpoints, columns=columns)
    
    # Separate data based on the flag
    df_lim = df[df['flag'] == 1].copy()
    df_flux = df[df['flag'] != 1].copy()
    
    # Calculate necessary values for df_lim
    df_lim['e_lim_min'] = df_lim['e'] - df_lim['e_min']
    df_lim['e_lim_max'] = df_lim['e_max'] - df_lim['e']
    df_lim['phi_lim'] = df_lim['e'] ** 2 * df_lim['flux']
    
    # Calculate necessary values for df_old
    df_flux['e_old_min'] = df_flux['e'] - df_flux['e_min']
    df_flux['e_old_max'] = df_flux['e_max'] - df_flux['e']
    df_flux['flux_old'] = df_flux['e'] ** 2 * df_flux['flux']
    df_flux['flux_old_min'] = df_flux['flux_old'] - df_flux['e'] ** 2 * df_flux['flux_min']
    df_flux['flux_old_max'] = df_flux['e'] ** 2 * df_flux['flux_max'] - df_flux['flux_old']

    return df_lim, df_flux
    


class SqueezedNorm(mpl.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, mid=0, s1=2, s2=2, clip=False):
        self.vmin = vmin # minimum value
        self.mid  = mid  # middle value
        self.vmax = vmax # maximum value
        self.s1=s1; self.s2=s2
        f = lambda x, zero,vmax,s: np.abs((x-zero)/(vmax-zero))**(1./s)*0.5
        self.g = lambda x, zero,vmin,vmax, s1,s2: f(x,zero,vmax,s1)*(x>=zero) - \
                                             f(x,zero,vmin,s2)*(x<zero)+0.5
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        r = self.g(value, self.mid,self.vmin,self.vmax, self.s1,self.s2)
        return np.ma.masked_array(r)


    


# Define the function to plot each parameter
def plot_parameter(ax, parameter, ref_model, comp_model, comp_val, comp_err, y_label, y_lim=None):
    labels = ['2019', 'OFF bkg', 'FOV']
    
    ax.errorbar(2, ref_model.parameters[parameter].value, yerr=ref_model.parameters[parameter].error, fmt='o', color='#8c96c6', capsize=5)
    ax.errorbar(3, comp_model.parameters[parameter].value, yerr=comp_model.parameters[parameter].error, fmt='o', color='#810f7c', capsize=5)
    ax.errorbar(1, comp_val, yerr=comp_err, fmt='o', color='#1d91c0', capsize=5)
    
    ax.set_ylabel(y_label)
    ax.set_xlim(0.5, 3.5)
    if y_lim:
        ax.set_ylim(y_lim)
    ax.set_xticks([1, 2, 3], labels)




# Define a function to calculate error bars
def calculate_error(value, error, min_val, max_val):
    yerr_neg = max(value - (min_val - error), error)
    yerr_pos = max((max_val + error) - value, error)
    return np.array([[yerr_neg, yerr_pos]]).T



# Define a function to plot each parameter
def plot_parameter_systematics(ax, parameter, ref_model, min_model, max_model, comp_val, comp_err, y_label, y_lim=None):
    
    labels = ['Reference', 'Matching',  'Template']

    value = ref_model.parameters[parameter].value
    error = ref_model.parameters[parameter].error
    min_val = min_model.parameters[parameter].value
    max_val = max_model.parameters[parameter].value
    
    ax.errorbar(2, value, yerr=calculate_error(value, error, min_val, max_val), fmt='o', color='red', capsize=5)
    ax.errorbar(2, value, yerr=error, fmt='o', color='#8c96c6', capsize=5)
    ax.errorbar(3, ref_model.parameters[parameter].value, yerr=ref_model.parameters[parameter].error, fmt='o', color='#810f7c', capsize=5)
    ax.errorbar(1, comp_val, yerr=comp_err, fmt='o', color='#1d91c0', capsize=5)
    
    ax.set_ylabel(y_label)
    ax.set_xlim(0.5, 3.5)
    if y_lim:
        ax.set_ylim(y_lim)
    ax.set_xticks([1, 2, 3], labels)




# Define a function to plot histograms and fitted Gaussian curves
def plot_significance_histograms(ax, bins, data_list, colors, colors_fit, labels):
    x_vals = 0.5 * (bins[:-1] + bins[1:])
    
    for i, data in enumerate(data_list):
        
        estimator = ExcessMapEstimator(correlation_radius="0.06 deg",)
        lima_maps = estimator.run(data)

        map = lima_maps["sqrt_ts"].data[np.isfinite(lima_maps["sqrt_ts"].data)]
        n, _, _ = ax.hist(map, bins=bins, histtype='step', color=colors[i], lw=1, zorder=1)
        mu, std = norm.fit(map)
        y = norm.pdf(x_vals, mu, std) * sum(n * np.diff(bins))
        ax.plot(x_vals, y, lw=2, color=colors_fit[i], label=r"{}: $\mu$ = {:.2f}, $\sigma$ = {:.2f}".format(labels[i], mu, std), zorder=2)
    
    ax.grid(ls='--')
    ax.legend(loc=1, fancybox=True, shadow=True)
    # ax.set_xlim(-6, 15)
    ax.set_ylim(10**0, 10**5)
    ax.set_ylabel('Counts')
    ax.set_xlabel('Significance')
    ax.set_yscale('log')





























