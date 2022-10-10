#%%
from pydoc import render_doc
from tkinter.messagebox import NO
from webbrowser import BackgroundBrowser
import numpy as np
import pandas as pd
import plotly.express as px
# from pyntcloud import PyntCloud
# from pythreejs import *
from graphnet.data.sqlite.sqlite_selection import (
    get_desired_event_numbers,
)
from graphnet.models.training.utils import (
    make_dataloader,
)
from graphnet.data.constants import FEATURES, TRUTH

features = FEATURES.DEEPCORE
#%%
truth = TRUTH.DEEPCORE[:-1]


config = {
        "db": "/groups/hep/aske/IceCube/storage/data/dev_lvl7_robustness_muon_neutrino_0000/data/dev_lvl7_robustness_muon_neutrino_0000.db",
        "pulsemap": "SRTTWOfflinePulsesDC",
        "batch_size": 512,
        "num_workers": 10,
        "accelerator": "gpu",
        "target": "energy",
    }

selection = get_desired_event_numbers(config["db"],100,0/5,2/5,1/5,1/5,1/5,42)
#%%
data_loader = make_dataloader(
        db = config["db"],
        selection = selection,
        pulsemaps = config["pulsemap"],
        features =features,
        truth = truth,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        )

print("hi")


dom_x = data_loader.dataset[0]['dom_x']
dom_y = data_loader.dataset[0]['dom_y']
dom_z = data_loader.dataset[0]['dom_z']
energy = data_loader.dataset[0]['charge']
#%%
points = pd.DataFrame(np.vstack([dom_x,dom_y,dom_z,energy]).T,columns=['x','y','z','charge'])

# %%

fig = px.scatter_3d(points, x='x', y='y', z='z',color='charge')
fig.show()

# %%
