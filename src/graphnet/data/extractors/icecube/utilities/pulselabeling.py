"""
Methods for Applying Pulse Labeling to RecoPulse Maps
"""

"""
Steps:
1. Label the MC Pulses
2. Label Lateral, Leading, Coincident, and Noise Hits in MCPulses 
3. Label the Reco Pulses using MCPulses
4. Return the labeled reco pulses
5. Done
6. Profit
"""

from collections import defaultdict 
import numpy as np
import pandas as pd
from icecube import phys_services
from icecube import dataclasses

def get_all_bundle_muons(frame):

    mc_tree_info = frame['I3MCTree']
    primaries = mc_tree_info.primaries
    bundle_particles = mc_tree_info.get_daughters(primaries[0])
    bundle_particle_ids = []

    for particle in bundle_particles:
        if (particle.type_string in ['MuPlus', 'MuMinus'] and particle.location_type_string == 'InIce'):
            bundle_particle_ids.append(particle)
    
    return bundle_particle_ids

def make_multiplicity_statistics(
    frame,
    event_pulses,
):
    
    full_mctree = frame['I3MCTree']
    primary = frame['PolyplopiaPrimary']

    bundle_particles = full_mctree.get_daughters(primary)
    bundle_muons = []
    for muon in bundle_particles:
        if (muon.type_string in ['MuPlus', 'MuMinus'] and muon.location_type_string == 'InIce'):
            bundle_muons.append(muon)
    
    muon_multiplicity_surface = len(bundle_muons)

    cylinder_muons = frame['MMCTrackList']
    tracks = []
    for track in cylinder_muons:
        if full_mctree.is_in_subtree(primary, track.particle) == True: # Cleaning Coincidence Hits
            tracks.append(track)

    muon_multiplicity_cylinder = len(tracks)
    
    deposited_muon_multiplicity = len(event_pulses['muon_id'].unique())
    primary_residual = event_pulses[(event_pulses['hit_type_primary'] == 'lateral') & (event_pulses['residual_primary'] < 0)]
    primary_residual_multiplicity = len(primary_residual['muon_id'].unique())
    del primary_residual
    leading_residual = event_pulses[(event_pulses['hit_type_energy'] == 'lateral') & (event_pulses['residual_energy'] < 0)]
    leading_residual_multiplicity = len(leading_residual['muon_id'].unique())
    del leading_residual
    charge_residual_multiplicity = event_pulses[(event_pulses['hit_type_charge'] == 'lateral') & (event_pulses['residual_charge'] < 0)]
    charge_residual_multiplicity = len(charge_residual_multiplicity['muon_id'].unique())
    dict_muon_mult = {
        'muon_multiplicity_surface': muon_multiplicity_surface,
        'muon_multiplicity_cylinder': muon_multiplicity_cylinder,
        'deposited_muon_multiplicity': deposited_muon_multiplicity,
        'primary_residual_multiplicity': primary_residual_multiplicity,
        'leading_residual_multiplicity': leading_residual_multiplicity,
        'charge_residual_multiplicity': charge_residual_multiplicity,
    }

    I3_double_container = dataclasses.I3MapStringDouble(dict_muon_mult)
    frame.Put('MultiplicityInfo', I3_double_container)

def get_muon_ids(frame):

    mc_tree = frame['I3MCTree']

    primaries = mc_tree.primaries
    bundle_particles = mc_tree.get_daughters(primaries[0])
    bundle_particle_ids = [0]
    bundle_particle_list = []
    photon_particle_ids = [0]

    for particle in bundle_particles:
        if (particle.type_string in ['MuPlus', 'MuMinus'] and particle.location_type_string == 'InIce'):
            if mc_tree.number_of_children(particle) > 0:
                first_child = mc_tree.first_child(particle)
                bundle_particle_ids.append(particle.minor_id)
                bundle_particle_list.append(particle)
                photon_particle_ids.append(first_child.minor_id)

   
    last_child = mc_tree.children(bundle_particle_list[-1])[-1]

    last_child_id = last_child.minor_id + 1

    photon_particle_ids.append(last_child_id)
    bundle_particle_ids.append(-1)

    return np.asarray(photon_particle_ids), np.asarray(bundle_particle_ids)


def make_labeled_pulses(
    frame,
    geo,
):
    
    doms_hit_by_different_particles  = frame['I3MCPulseSeriesMapParticleIDMap']
    dom_pulses = frame['I3MCPulseSeriesMap']
    mc_tree_info = frame['I3MCTree']

    bundle_muons = get_all_bundle_muons(frame)
    photon_ids, muon_ids = get_muon_ids(frame)
    #print(photon_ids[-1])
    oms = defaultdict(list)

    for _, dom in enumerate(doms_hit_by_different_particles):

        #print(_, len(doms_hit_by_different_particles))
        dom_specific_pulses = np.asarray(dom_pulses[dom[0]])
        #print(_, len(dom_specific_pulses))

        dom_x = geo[dom[0]].position.x
        dom_y = geo[dom[0]].position.y
        dom_z = geo[dom[0]].position.z

        for particle in dom[1].keys():
            dom_hits = dom[1][particle]
            particle_specific_pulses = dom_specific_pulses[dom_hits]

            # Old Structure
            #for muon in bundle_muons:
            #    if mc_tree_info.is_in_subtree(muon, particle):
            #        muon_id = muon.minor_id
            #        break
            
            muon_id = 0
            minor_id = particle.minorID
            original_id = particle.minorID
            while minor_id >= photon_ids[-1]:
                particle = mc_tree_info.parent(particle)
                minor_id = particle.minor_id
               
            for _,pulse in enumerate(particle_specific_pulses):
                oms['time'].append(pulse.time)
                oms['charge'].append(pulse.charge)
                oms['x'].append(dom_x)
                oms['y'].append(dom_y)
                oms['z'].append(dom_z)
                oms['DOM'].append(dom[0])
                oms['string'].append(dom[0].string)
                oms['dom'].append(dom[0].om)
                oms['dom_hit'].append(dom_hits[_])
                oms['muon_id'].append(muon_id)
                oms['subbundle_id'].append(minor_id)
                oms['original_id'].append(original_id)



    labeled_pulses = pd.DataFrame(
        {
        'x': oms['x'], 
        'y': oms['y'], 
        'z': oms['z'], 
        "string":oms['string'], 
        "dom":oms['dom'],
        't':oms["time"], 
        'DOM': oms['DOM'],
        'charge':oms['charge'],
        'muon_id': oms['muon_id'],
        'subbundle_id': oms['subbundle_id'],
        'original_id': oms['original_id'],
        'dom_hit': oms['dom_hit'],
        },
    )

    muon_positions = np.searchsorted(photon_ids, labeled_pulses['subbundle_id'].to_numpy(), side='right')
    idx = np.maximum(muon_positions-1, 0)
    muon_labels = muon_ids[idx]

    labeled_pulses['muon_id'] = muon_labels

    return labeled_pulses


def add_pulse_labels(
    event_pulses,
    leading,
):

    event_pulses['hit_type_charge'] = 'unknown'
    event_pulses.loc[event_pulses['original_id'] == 0, 'hit_type_charge'] = 'noise'
    event_pulses.loc[event_pulses['muon_id'] == leading[0].minor_id, 'hit_type_charge'] = 'leading'
    event_pulses.loc[(event_pulses['original_id'] != 0) & (event_pulses['muon_id'] == 0), 'hit_type_charge'] = 'coincidence'
    event_pulses.loc[(event_pulses['muon_id'] != 0) & (event_pulses['muon_id'] != leading[0].minor_id), 'hit_type_charge'] = 'lateral'   

    event_pulses['hit_type_energy'] = 'unknown'
    event_pulses.loc[event_pulses['original_id'] == 0, 'hit_type_energy'] = 'noise'
    event_pulses.loc[event_pulses['muon_id'] == leading[1].minor_id, 'hit_type_energy'] = 'leading'
    event_pulses.loc[(event_pulses['original_id'] != 0) & (event_pulses['muon_id'] == 0), 'hit_type_energy'] = 'coincidence'
    event_pulses.loc[(event_pulses['muon_id'] != 0) & (event_pulses['muon_id'] != leading[1].minor_id), 'hit_type_energy'] = 'lateral'     
    
    event_pulses['hit_type_primary'] = 'unknown'
    event_pulses.loc[event_pulses['original_id'] == 0, 'hit_type_primary'] = 'noise'
    event_pulses.loc[event_pulses['muon_id'] == leading[2].minor_id, 'hit_type_primary'] = 'leading'
    event_pulses.loc[(event_pulses['original_id'] != 0) & (event_pulses['muon_id'] == 0), 'hit_type_primary'] = 'coincidence'
    event_pulses.loc[(event_pulses['muon_id'] != 0) & (event_pulses['muon_id'] != leading[2].minor_id), 'hit_type_primary'] = 'lateral' 

    return event_pulses

def compute_residual_information(
    frame,
    event_pulses,
    geo,
    leading,
):  
    
    event_pulses = add_pulse_labels(
        event_pulses=event_pulses,
        leading=leading,
    )

    doms_hit_by_different_particles  = frame['I3MCPulseSeriesMapParticleIDMap']
    dom_pulses = frame['I3MCPulseSeriesMap']

    bundle_muons = get_all_bundle_muons(frame)

    oms = defaultdict(list)

    for _, dom in enumerate(doms_hit_by_different_particles):
        
        dom_specific_pulses = np.asarray(dom_pulses[dom[0]])
        #print(_, len(doms_hit_by_different_particles))

        dom_x = geo[dom[0]].position.x
        dom_y = geo[dom[0]].position.y
        dom_z = geo[dom[0]].position.z

        for particle in dom[1].keys():
            particle_specific_pulses = dom_specific_pulses[dom[1][particle]]


            r = phys_services.I3Calculator.closest_approach_distance(leading[0], geo[dom[0]].position)
            r_energy = phys_services.I3Calculator.closest_approach_distance(leading[1], geo[dom[0]].position)
            r_primary = phys_services.I3Calculator.closest_approach_distance(leading[2], geo[dom[0]].position)

            t_charge = leading[0].time + phys_services.I3Calculator.cherenkov_time(leading[0],geo[dom[0]].position)
            t_energy = leading[1].time + phys_services.I3Calculator.cherenkov_time(leading[1],geo[dom[0]].position)
            t_primary = leading[2].time + phys_services.I3Calculator.cherenkov_time(leading[2],geo[dom[0]].position) 

            for _,pulse in enumerate(particle_specific_pulses):
                oms['time'].append(pulse.time)
                oms['charge'].append(pulse.charge)
                oms['x'].append(dom_x)
                oms['y'].append(dom_y)
                oms['z'].append(dom_z)
                oms['DOM'].append(dom[0])
                oms['string'].append(dom[0].string)
                oms['dom'].append(dom[0].om)
                oms['dom_hit'].append(dom[1][particle][_])
                oms['r'].append(r)
                oms['r_energy'].append(r_energy)
                oms['r_primary'].append(r_primary)
                oms['timing_residual'].append(pulse.time - t_charge)
                oms['timing_residual_energy'].append(pulse.time - t_energy)
                oms['timing_residual_primary'].append(pulse.time - t_primary)

    labeled_pulses = pd.DataFrame(
        {
        'x': oms['x'], 
        'y': oms['y'], 
        'z': oms['z'], 
        "string":oms['string'], 
        "dom":oms['dom'],
        't':oms["time"], 
        'DOM': oms['DOM'],
        'charge':oms['charge'],
        'dom_hit': oms['dom_hit'],
        'r':oms['r'],
        'r_energy':oms['r_energy'],
        'r_primary':oms['r_primary'],
        'residual': oms['timing_residual'],
        'residual_energy': oms['timing_residual_energy'],
        'residual_primary': oms['timing_residual_primary'],
        },
    )

    event_pulses['r'] = labeled_pulses['r']
    event_pulses['r_energy'] = labeled_pulses['r_energy']
    event_pulses['r_primary'] = labeled_pulses['r_primary']
    event_pulses['residual_charge'] = labeled_pulses['residual']
    event_pulses['residual_energy'] = labeled_pulses['residual_energy']
    event_pulses['residual_primary'] = labeled_pulses['residual_primary']

    return event_pulses

def get_leading_muon_charge(
    bundle_muons,
    pulses,
):

    pulses_only_labled = pulses[pulses['muon_id'] != 0]
    max_muon = pulses_only_labled.groupby('muon_id')['charge'].sum().idxmax()

    leading_charge = next(muon for muon in bundle_muons if muon.minor_id == max_muon)
    
    return leading_charge


def cleaned_charge(
    frame,
    reco_pulses,

    ):
    """
    Compute Homogenized QTot from Only Signal and Noise Events with Coincidence Events Removed
    -> Remove Coinicident Events
    -> Remove Duplicate Pulses
    -> Compute Total Charge of Event
    """


    remove_deepcore = reco_pulses[(reco_pulses['string'] < 79) & (reco_pulses['rde'] <= 1.1)]

    no_coindidence = remove_deepcore[remove_deepcore['hit_type_leading'] != 'coincidence']

    total_charge = remove_deepcore['charge'].sum()

    q_total = no_coindidence.groupby(["string", "dom"], as_index=False)['charge'].sum()

    q_total_bubble_cut = q_total[q_total['charge'] < total_charge/2]

    frame['HQTOT_NO_COINCIDENCE'] = dataclasses.I3Double(q_total_bubble_cut.charge.sum())


def apply_label(
    row,
    df2,
    hit_type,
):
    
    selected = df2[(df2['string'].to_numpy() == row['string']) & (df2['dom'].to_numpy()  == row['dom_number'])& (np.abs(df2['t'].to_numpy()  - row['time']) < 25)][hit_type]
    
    if 'leading' in selected.values:
        return 'leading'
    elif 'lateral' in selected.values:
        return 'lateral'
    elif 'coincidence' in selected.values:
        return 'coincidence'
    else:
        return 'noise'

def label_reco_pulses_newer(
    reco_pulses,
    mc_pulses,
):
    
    reco_pulses['hit_type_charge'] = reco_pulses.apply(apply_label, axis=1, df2=mc_pulses[['string', 'dom', 't', 'hit_type_charge']], hit_type='hit_type_charge')
    reco_pulses['hit_type_energy'] = reco_pulses.apply(apply_label, axis=1, df2=mc_pulses[['string', 'dom', 't', 'hit_type_energy']], hit_type='hit_type_energy')
    reco_pulses['hit_type_primary'] = reco_pulses.apply(apply_label, axis=1, df2=mc_pulses[['string', 'dom', 't', 'hit_type_primary']], hit_type='hit_type_primary')

    return reco_pulses


def information_from_first_hit(
    frame,
    leading_muon_type,
    pulses,
):
    """
    Information about Number of Residual Hits in Early Time Window Hits
    -> The Time is the Number of Hits indexed from the First Hit on a DOM
    """

    pulses = pulses[pulses['dom_hit'] == 0]

    leading_hits = pulses[f'hit_type_{leading_muon_type}'] == 'leading'
    lateral_hits = pulses[f'hit_type_{leading_muon_type}'] == 'lateral'
    noise_hits = pulses[f'hit_type_{leading_muon_type}'] == 'noise'
    residual_hits = pulses[f'residual_{leading_muon_type}'] < 0
    non_residual_hits = pulses[f'residual_{leading_muon_type}'] >= 0

    dict_info = {
        'leading': len(pulses[leading_hits]),
        'lateral': len(pulses[lateral_hits]),
        'noise': len(pulses[noise_hits]),
        'leading_pos': len(pulses[leading_hits | (lateral_hits & non_residual_hits)]),
        'lateral_neg': len(pulses[lateral_hits & residual_hits]),
    }

    if leading_muon_type == 'hit_type':
        leading_muon = 'hit_type_charge'
    else:
        leading_muon = leading_muon_type

    I3_double_container = dataclasses.I3MapStringDouble(dict_info)
    frame.Put(f'FirstHitInfo_{leading_muon}', I3_double_container)


def information_within_time(
    frame,
    leading_muon_type,
    time,
    pulses,
):
    """
    Information about Number of Residual Hits in Early Time Window Hits
    -> The Time is the Number of Hits indexed from the First Hit on a DOM
    """
    earliest_hits = pulses.groupby(['string', 'dom_number'])['time'].transform('min')

    dict_info = {}
    only_early_hits = pulses[(pulses['time'] - earliest_hits) < time]

    leading_hits = only_early_hits[f'hit_type_{leading_muon_type}'] == 'leading'
    lateral_hits = only_early_hits[f'hit_type_{leading_muon_type}'] == 'lateral'
    noise_hits = only_early_hits[f'hit_type_{leading_muon_type}'] == 'noise'
    residual_hits = only_early_hits[f'residual_{leading_muon_type}'] < 0
    non_residual_hits = only_early_hits[f'residual_{leading_muon_type}'] >= 0

    new_info = {
        f'leading_{time}': len(only_early_hits[leading_hits]),
        f'lateral_{time}': len(only_early_hits[lateral_hits]),
        f'noise_{time}': len(only_early_hits[noise_hits]),
        f'leading_pos_{time}': len(only_early_hits[leading_hits | (lateral_hits & non_residual_hits)]),
        f'lateral_neg_{time}': len(only_early_hits[lateral_hits & residual_hits]),
    }

    dict_info.update(new_info)


    I3_double_container = dataclasses.I3MapStringDouble(dict_info)

    frame.Put(f'ResidualTimeInfo_{leading_muon_type}', I3_double_container)

def information_from_total_charge(
    frame,
    leading_muon_type,
    charge,
    pulses,
):
    """
    Summary of lateral muon information deposited before a certain charge deposited
    """
    
    """
    Information about Number of Residual Hits in Early Time Window Hits
    -> The Time is the Number of Hits indexed from the First Hit on a DOM
    """
    pulses = pulses.sort_values(['string', 'dom_number', 'time'])
    pulses['charge_sum'] = pulses.groupby(['string', 'dom_number'])['charge'].cumsum()

    only_early_hits = pulses[pulses['charge_sum'] < charge]

    leading_hits = only_early_hits[f'hit_type_{leading_muon_type}'] == 'leading'
    lateral_hits = only_early_hits[f'hit_type_{leading_muon_type}'] == 'lateral'
    noise_hits = only_early_hits[f'hit_type_{leading_muon_type}'] == 'noise'
    residual_hits = only_early_hits[f'residual_{leading_muon_type}'] < 0
    non_residual_hits = only_early_hits[f'residual_{leading_muon_type}'] >= 0

    dict_info = {
        f'leading_{charge}': len(only_early_hits[leading_hits]),
        f'lateral_{charge}': len(only_early_hits[lateral_hits]),
        f'noise_{charge}': len(only_early_hits[noise_hits]),
        f'leading_pos_{charge}': len(only_early_hits[leading_hits | (lateral_hits & non_residual_hits)]),
        f'lateral_neg_{charge}': len(only_early_hits[lateral_hits & residual_hits]),
    }

    I3_double_container = dataclasses.I3MapStringDouble(dict_info)

    frame.Put(f'ResidualChargeInfo_{leading_muon_type}', I3_double_container)