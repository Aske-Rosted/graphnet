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
import polars as pl

from graphnet.utilities.imports import has_icecube_package

if has_icecube_package():
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
    event_pulses: pl.DataFrame,
):
    """
    Takes a labeled pulsemap and labels the multiplicity of muons
    """
    
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
        #if full_mctree.is_in_subtree(primary, track.particle) == True: # Cleaning Coincidence Hits
        tracks.append(track)

    muon_multiplicity_cylinder = len(tracks)
    
    deposited_muon_multiplicity = (
        event_pulses.filter(pl.col('hit_type_primary') == 'lateral')
        .select(pl.col('muon_id').n_unique())
        .item()
    )

    primary_residual_multiplicity = (
        event_pulses.filter(
            (pl.col('hit_type_primary') == 'lateral') & (pl.col('residual_primary') < 0)
        )
        .select(pl.col('muon_id').n_unique())
        .item()
    )

    leading_residual_multiplicity = (
        event_pulses.filter(
            (pl.col('hit_type_energy') == 'lateral') & (pl.col('residual_energy') < 0)
        )
        .select(pl.col('muon_id').n_unique())
        .item()
    )

    charge_residual_multiplicity = (
        event_pulses.filter(
            (pl.col('hit_type_charge') == 'lateral') & (pl.col('residual_charge') < 0)
        )
        .select(pl.col('muon_id').n_unique())
        .item()
    )

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

    photon_ids, muon_ids = get_muon_ids(frame)
    oms = defaultdict(list)

    for _, dom in enumerate(doms_hit_by_different_particles):

        dom_specific_pulses = np.asarray(dom_pulses[dom[0]])

        dom_x = geo[dom[0]].position.x
        dom_y = geo[dom[0]].position.y
        dom_z = geo[dom[0]].position.z

        for particle in dom[1].keys():
            dom_hits = dom[1][particle]
            particle_specific_pulses = dom_specific_pulses[dom_hits]
            
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

    labeled_pulses = pl.DataFrame(
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

    labeled_pulses.with_columns(
        muon_id = pl.Series(muon_labels)
    )

    return labeled_pulses

def add_pulse_labels(
    event_pulses: pl.DataFrame,
    leading: list,
):
    
    def make_label(muon_id):
        return (
            pl.when(pl.col('original_id') == 0).then(pl.lit('noise'))
            .when(pl.col('muon_id') == muon_id).then(pl.lit('leading'))
            .when((pl.col('original_id') != 0) & (pl.col('muon_id') == 0)).then(pl.lit('coincidence'))
            .when((pl.col('muon_id') != 0) & (pl.col('muon_id') != muon_id)).then(pl.lit('lateral'))
            .otherwise(pl.lit('unknown'))
        )
    
    return event_pulses.with_columns([
        make_label(leading[0].minor_id).alias('hit_type_charge'),
        make_label(leading[1].minor_id).alias('hit_type_energy'),
        make_label(leading[2].minor_id).alias('hit_type_primary'),
    ])

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

    labeled_pulses = pl.DataFrame(
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

    event_pulses = event_pulses.with_columns([
        pl.Series("r", labeled_pulses["r"]),
        pl.Series("r_energy", labeled_pulses["r_energy"]),
        pl.Series("r_primary", labeled_pulses["r_primary"]),
        pl.Series("residual_charge", labeled_pulses["residual"]),
        pl.Series("residual_energy", labeled_pulses["residual_energy"]),
        pl.Series("residual_primary", labeled_pulses["residual_primary"]),
    ])

    return event_pulses

def get_leading_muon_charge(
    bundle_muons,
    pulses,
):

    max_muon = (
        pulses.filter(pl.col('muon_id') != 0).groupby('muon_id')
        .agg(pl.col('charge').sum())
        .sort('charge', descending=True)
        .select(pl.col('muon_id').first())
        .item()
    )

    leading_charge = next(muon for muon in bundle_muons if muon.minor_id == max_muon)
    
    return leading_charge


def cleaned_charge(
    frame,
    reco_pulses: pl.DataFrame,

    ):
    """
    Compute Homogenized QTot from Only Signal and Noise Events with Coincidence Events Removed
    -> Remove Coinicident Events
    -> Remove Duplicate Pulses
    -> Compute Total Charge of Event
    """

    homogenized_charge_filter = reco_pulses.filter(
        (pl.col('string') < 79) & (pl.col('rde') <= 1.1) &
        (pl.col('hit_type_leading') != 'coincidence')
    )

    total_charge = homogenized_charge_filter.select(pl.col('charge').sum()).item()

    q_total_bubble_cut_sum = (
        homogenized_charge_filter.groupby(['string', 'dom'])
        .agg(pl.col('charge').sum())
        .filter(pl.col('charge') < total_charge/2)
        .select(pl.col('charge').sum())
        .item() or 0.0
    )

    frame['HQTOT_NO_COINCIDENCE'] = dataclasses.I3Double(q_total_bubble_cut_sum)

def label_reco_pulses(
    reco_pulses: pl.DataFrame,
    mc_pulses: pl.DataFrame,
):
    reco_pulses = reco_pulses.with_row_index("pulse_id")
    
    mc_subset = mc_pulses.select([
        pl.col('string'),
        pl.col('dom').alias('dom_number'),
        pl.col('t').alias('mc_time'),
        pl.col('hit_type_charge'),
        pl.col('hit_type_energy'),
        pl.col('hit_type_primary'),
    ])

    combined = reco_pulses.join(mc_subset, on=['string', 'dom_number'], how='left')

    def determine_hit_type(col_name):
        return (
            pl.when(pl.col('mc_time').is_null()).then(pl.lit('noise'))
            .when((pl.col('time') - pl.col('mc_time')).abs() < 25).then(pl.col(col_name))
            .otherwise(pl.lit('noise'))
            .sort_by(
                pl.when(pl.col(col_name) == 'leading').then(0)
                .when(pl.col(col_name) == 'lateral').then(1)
                .when(pl.col(col_name) == 'coincidence').then(2)
                .otherwise(3)
            )
            .first()
            .over(pl.col('pulse_id'))
        )

    labeled_pulses = combined.with_columns(
        determine_hit_type('hit_type_charge').alias('hit_type_charge'),
        determine_hit_type('hit_type_energy').alias('hit_type_energy'),
        determine_hit_type('hit_type_primary').alias('hit_type_primary'),
    )

    return labeled_pulses.unique(subset=['pulse_id']).drop('pulse_id')

