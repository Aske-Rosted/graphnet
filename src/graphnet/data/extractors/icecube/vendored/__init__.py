"""I3Extractors vendored from other GraphNeT forks, for deployment only.

The pulse extractors of Maxwell Nakos' production bundle models (see
README.md in `graphnet/models/vendored`). Exported from
`graphnet.data.extractors.icecube` under their original names.
"""

from .i3pulseextractornew import (
    I3PulseExtractorNew,
    I3PulseExtractorNewIceCube86,
)
from .i3inferencepulseextractornew import (
    I3InferencePulseExtractorNew,
    I3InferencePulseExtractorNewIceCube86,
)
