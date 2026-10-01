"""Physics task-specific modules to be used as model "read-outs"."""

from .task import (
    Task,
    IdentityTask,
    IdentityTaskWithUncertainty,
    StandardLearnedTask,
    StandardFlowTask,
)
from .heads import TaskHead, MLPHead
