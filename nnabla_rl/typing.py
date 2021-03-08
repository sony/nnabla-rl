from typing import Any, Dict, Tuple, Type

import numpy as np

State = Type[np.array]
Action = Type[np.array]
Reward = float
NonTerminal = float
NextState = Type[np.array]
Info = Dict[str, Any]
Experience = Tuple[State, Action, Reward, NonTerminal, NextState, Info]
