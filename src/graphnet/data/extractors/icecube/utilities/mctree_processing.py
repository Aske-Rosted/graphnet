from icecube import icetray, dataio
from icecube import dataclasses
from icecube import phys_services
from icecube import dataclasses, simclasses, icetray
from icecube import MuonGun, simclasses
import numpy as np
import pandas as pd

from collections import defaultdict 
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.pyplot as plt


from graphnet.data.extractors.icecube.utilities.vector_computations import closest_approach_distance_vector

def get_leading_particle(
    frame,
):
    
    primary = frame['PolyplopiaPrimary']
    pdg = frame['PolyplopiaPrimary'].pdg_encoding
    full_mctree = frame['I3MCTree']
    mctree = frame['I3MCTree_preMuonProp']
    if np.abs(pdg) in [12, 14, 16]:
        current = mctree[1]
        while mctree.number_of_children(current) > 0:
            current = mctree.first_child(current)

        primary.pos.x = current.pos.x
        primary.pos.y = current.pos.y
        primary.pos.z = current.pos.z
    else:
        current = mctree[frame['PolyplopiaPrimary']]
        highest_energy = -1
        bundle_particles = mctree.get_daughters(current)
        for particle in bundle_particles:
            if (particle.type_string in ['MuPlus', 'MuMinus'] and particle.location_type_string == 'InIce'):
                if particle.energy > highest_energy:
                    highest_energy = particle.energy
                    current = particle

    tracklist = frame['MMCTrackList']

    e_initial = 0
    for track in tracklist:
        if track.Ei > e_initial:
            e_initial = track.Ei
            current = track.particle
    
    return primary, current

def turn_mctree_into_light_sources(
    frame,
):
    """
    Take the Input MCTree and output a dataframe of energy deposits
    -> Position (x,y,z)
    -> Energy
    -> Distance from the Primary Track (radius)
    -> Point Along the Track (Relative to Closest Approach Point to the Detector)
    """
    
    
    mctree = frame['I3MCTree']
    primary = frame['PolyplopiaPrimary']
    number_primaries = len(mctree.get_primaries())
    oms = defaultdict(list)
        
    """
    Calculating Muon Losses Correctly
    -> Exclude Muons (Keep track of total muon energy at a position, not the losses)
    -> PairProd that have children are dark, so they are excluding
    -> PairProd that don't have childer are null, so don't exclude null particles
    -> Exclude Primary Particles
    -> Exclude other dark particles
    -> Exclude Neutrinos
    -> Hadrons with Children are dark, already excluded
    -> Compute Continous Losses by Taking the Difference Between Muon Energies and Subtracting the Stochastic Losses
    """

    for particle in mctree:
        
        if particle.location_type != dataclasses.I3Particle.LocationType.InIce:
            continue
        if particle.shape == dataclasses.I3Particle.ParticleShape.Dark:
            continue
        if particle.shape == dataclasses.I3Particle.ParticleShape.Primary:
            continue
        if (np.abs(particle.pdg_encoding) == 12) | (np.abs(particle.pdg_encoding) == 14) | (np.abs(particle.pdg_encoding) == 16):
            continue
        if number_primaries > 1:
            if mctree.get_primary(particle).minor_id != primary.minor_id: # Cleaning Coincidence Hits
            #if mctree.is_in_subtree(primary, particle) == False: # Cleaning Coincidence Hits
                continue
        if np.abs(particle.pdg_encoding) in [13,15]:  # Selecting Muons and Taus
            try:
                previous_sibling = mctree.previous_sibling(particle)
                stochastic_losses = []
                while (np.abs(previous_sibling.pdg_encoding) not in [13,15]) or (previous_sibling.shape == dataclasses.I3Particle.ParticleShape.Dark):
                    if previous_sibling.shape != dataclasses.I3Particle.ParticleShape.Dark:
                        stochastic_losses.append(previous_sibling)
                    try:
                        previous_sibling = mctree.previous_sibling(previous_sibling)
                    except:
                        break
                try:
                    next_sibling = mctree.next_sibling(particle)
                    if next_sibling.minor_id < particle.minor_id:
                        if next_sibling.shape != dataclasses.I3Particle.ParticleShape.Dark:
                            stochastic_losses.append(next_sibling)
                except:
                    pass
                
                stoch_loss = 0
                for loss in stochastic_losses:
                    if loss.minor_id > previous_sibling.minor_id:
                        stoch_loss += loss.energy

                cont_loss = previous_sibling.energy - stoch_loss - particle.energy
                loss_pos = previous_sibling.pos + (particle.pos - previous_sibling.pos)/2
                oms['x'].append(loss_pos.x)
                oms['y'].append(loss_pos.y)
                oms['z'].append(loss_pos.z)
                oms['energy'].append(cont_loss)

            except:
                # Last Particle
                pass
            continue # Only Adding the Coninuous Losses
              
        # Only add particles if the particles are a light source in the detector
        oms['x'].append(particle.pos.x)
        oms['y'].append(particle.pos.y)
        oms['z'].append(particle.pos.z)
        oms['energy'].append(particle.energy)
    
    mctree_information = pd.DataFrame(oms)    

    return mctree_information

def turn_mcpe_into_light_sources(
    frame,
):
    """
    Take the Input MCPE Series and output a dataframe of energy deposits
    -> Position (x,y,z)
    -> Energy
    -> Distance from the Primary Track (radius)
    -> Point Along the Track (Relative to Closest Approach Point to the Detector)
    -> Number of MCPEs associated with the light source
    """
    
    
    doms_hit_by_different_particles  = frame['I3MCPESeriesMapParticleIDMap']
    try:
        dom_pulses = frame['I3MCPESeriesMapWithoutNoise']
    except:
        dom_pulses = frame['I3MCPESeriesMap']

    oms = defaultdict(list)
    mc_tree = frame['I3MCTree']
    primary = frame['PolyplopiaPrimary']
    number_primaries = len(mc_tree.get_primaries())

    coincidence_hits = 0
    for _, dom in enumerate(doms_hit_by_different_particles):

        for particle in dom[1].keys():
            dom_hits = dom[1][particle]

            full_particle = mc_tree[particle]
            # Skip If Noise Hits
            if (particle.minorID == 0) and (particle.majorID == 0):
                continue
            if number_primaries > 1:
                if mc_tree.get_primary(particle).minor_id != primary.minor_id: # Cleaning Coincidence Hits
                #if mc_tree.is_in_subtree(primary, full_particle) == False: # Cleaning Coincidence Hits
                    coincidence_hits += len(dom_hits)
                    continue
        
            oms['x'].append(full_particle.pos.x)
            oms['y'].append(full_particle.pos.y)
            oms['z'].append(full_particle.pos.z)
            oms['MCPEs'].append(len(dom_hits))
    
    
    mcpe_information = pd.DataFrame(oms)
    if len(mcpe_information) == 0:
        frame['fraction_coincidence'] = dataclasses.I3Double(1)
    else:
        frame['fraction_coincidence'] = dataclasses.I3Double(coincidence_hits / (mcpe_information['MCPEs'].sum() + coincidence_hits))

    return mcpe_information
    

def add_closest_approach_vectors(
    leading,
    pulses,
):
    
    r, cx, cy, cz, s = closest_approach_distance_vector(
        leading,
        pulses['x'],
        pulses['y'],
        pulses['z'],
    )

    dr, dcx, dcy, dcz, ds = closest_approach_distance_vector(
        leading,
        0,
        0,
        0,
    )

    pulses['r'] = r
    # Closest Approach Point to the Detector Origin is Zero
    pulses['s'] = s - ds
    pulses['r_det'] = np.sqrt(pulses['x']**2 + pulses['y']**2)
    return pulses

def compute_training_labels(
    frame,
    light_source_information,
    information_type,
    max_s,
    min_s,
    leading_type = 'Primary',
):
    """
    Compute the Training Labels for the Light Source Information
    """

    r = light_source_information['r']
    s = light_source_information['s']
    w = light_source_information[information_type]
    
    bin_count = 100
    """
    Stochasticity
    """
    dict_training = {}
    try:
        hist, edges = np.histogram(s, bins = np.linspace(-1200,1200, bin_count), weights = w)
        # Exclude Histogram Bins outside of max and min s       
        hist = hist[(edges[1:] > min_s) & (edges[:-1] < max_s)]
        std = np.std(hist)
        peak_above_mean = np.max(hist) - np.mean(hist)
        peak_above_meadian = np.max(hist) - np.median(hist)

        dict_training.update({
            f'stochasticity_std': std,
            f'stochasticity_peak_above_mean': peak_above_mean,
            f'stochasticity_peak_above_median': peak_above_meadian,
            f'stochasticity_ratio_above_mean': np.max(hist) / np.mean(hist),
            f'stochasticity_ratio_above_median': np.max(hist) / np.median(hist),
        })
    except:
        dict_training.update({
            f'stochasticity_std': -1,
            f'stochasticity_peak_above_mean': -1,
            f'stochasticity_peak_above_median': -1,
            f'stochasticity_ratio_above_mean': -1,
            f'stochasticity_ratio_above_median': -1,
        })
        

    """
    Lateral Distribution: Weighted RMS
    """

    mask = (s >= min_s) & (s <= max_s)
    rms = np.sqrt(np.sum(w[mask] * r[mask]**2) / np.sum(w[mask]))
    rms3 = np.sqrt(np.sum(w[mask] * r[mask]**3) / np.sum(w[mask]))
    rms4 = np.sqrt(np.sum(w[mask] * r[mask]**4) / np.sum(w[mask]))
    most_lateral_deposit = np.max(r[mask])

    try:
        mask = (s >= min_s) & (s <= max_s)
        rms = np.sqrt(np.sum(w[mask] * r[mask]**2) / np.sum(w[mask]))
        rms3 = np.sqrt(np.sum(w[mask] * r[mask]**3) / np.sum(w[mask]))
        rms4 = np.sqrt(np.sum(w[mask] * r[mask]**4) / np.sum(w[mask]))
        most_lateral_deposit = np.max(r[mask])
    except:
        rms = -1
        rms3 = -1
        rms4 = -1
        most_lateral_deposit = -1
        
    dict_training.update({
        'lateral_rms': rms,
        'lateral_rms3': rms3,
        'lateral_rms4': rms4,
        'most_lateral_deposit': most_lateral_deposit,
    })

    frame[f'{leading_type}ShowerProfile_{information_type}'] = dataclasses.I3MapStringDouble(dict_training)

def no_signal_hits(
    frame,
    information_type,
    leading_type = 'Primary'
):
    """
    Stochasticity
    """
    dict_training = {}
        
    dict_training.update({
        f'stochasticity_std': -1,
        f'stochasticity_peak_above_mean': -1,
        f'stochasticity_peak_above_median': -1,
        f'stochasticity_ratio_above_mean': -1,
        f'stochasticity_ratio_above_median': -1,
    })
        

    """
    Lateral Distribution: Weighted RMS
    """

    rms = -1
    rms3 = -1
    rms4 = -1
    most_lateral_deposit = -1
        
    dict_training.update({
        'lateral_rms': rms,
        'lateral_rms3': rms3,
        'lateral_rms4': rms4,
        'most_lateral_deposit': most_lateral_deposit,
    })

    frame[f'{leading_type}ShowerProfile_{information_type}'] = dataclasses.I3MapStringDouble(dict_training)

def make_shower_and_stochasticity_info(
    frame,
):
    
    mctree_information = turn_mctree_into_light_sources(
        frame,
    )

    mcpe_information = turn_mcpe_into_light_sources(
        frame,
    )
    
    if len(mcpe_information) <= 5:
        print('no signal hits', frame['I3EventHeader'].event_id)
        no_signal_hits(
            frame,
            'MCPEs',
            leading_type='Primary'
        )

        no_signal_hits(
            frame,
            'energy',
            leading_type='Primary'
        )

        no_signal_hits(
            frame,
            'MCPEs',
            leading_type='Leading'
        )

        no_signal_hits(
            frame,
            'energy',
            leading_type='Leading'
        )


        length_deposited = {
            'primary_length_deposited': -1,
            'leading_length_deposited': -1,
        }

        frame['ShowerLengthDeposited'] = dataclasses.I3MapStringDouble(length_deposited)
        

        return
    
    primary, leading_particle = get_leading_particle(frame)
    primary = frame['PolyplopiaPrimary']
    primary_mctree_information = add_closest_approach_vectors(
        leading=primary,
        pulses=mctree_information.copy(),
    )

    leading_mctree_information = add_closest_approach_vectors(
        leading=leading_particle,
        pulses=mctree_information.copy(),
    )

    del mctree_information

    primary_mcpe_information = add_closest_approach_vectors(
        leading=primary,
        pulses=mcpe_information.copy(),
    )

    leading_mcpe_information = add_closest_approach_vectors(
        leading=leading_particle,
        pulses=mcpe_information.copy(),
    )

    del mcpe_information
    
    # Min S and Max S
    s_min_primary = np.min(primary_mcpe_information['s'])
    s_max_primary = np.max(primary_mcpe_information['s'])
    s_min_leading = np.min(leading_mcpe_information['s'])
    s_max_leading = np.max(leading_mcpe_information['s'])

    length_deposited = {
        'primary_length_deposited': s_max_primary-s_min_primary,
        'leading_length_deposited': s_max_leading-s_min_leading,
    }

    frame['ShowerLengthDeposited'] = dataclasses.I3MapStringDouble(length_deposited)
    
    compute_training_labels(
        frame,
        primary_mcpe_information,
        'MCPEs',
        max_s = s_max_primary,
        min_s = s_min_primary,
        leading_type='Primary'
    )

    del primary_mcpe_information

    compute_training_labels(
        frame,
        leading_mcpe_information,
        'MCPEs',
        max_s = s_max_leading,
        min_s = s_min_leading,
        leading_type='Leading'
    )

    del leading_mcpe_information

    compute_training_labels(
        frame,
        primary_mctree_information,
        'energy',
        max_s = s_max_primary,
        min_s = s_min_primary,
        leading_type='Primary'
    )

    del primary_mctree_information

    compute_training_labels(
        frame,
        leading_mctree_information,
        'energy',
        max_s = s_max_leading,
        min_s = s_min_leading,
        leading_type='Leading'
    )

    del leading_mctree_information