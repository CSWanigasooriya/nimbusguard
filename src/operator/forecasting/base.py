
from abc import ABC, abstractmethod
from typing import Dict, Optional

import pandas as pd


class BasePredictor(ABC):
    """Abstract base class for all time series predictors."""

    @abstractmethod
    async def predict(
        self, historical_data: Dict[str, pd.DataFrame]
    ) -> Optional[Dict[str, float]]:
        """
        Generate a forecast based on historical data.
        
        Args:
            historical_data: A dictionary where keys are feature names and
                             values are pandas DataFrames with 'timestamp' and 'value'.
                             
        Returns:
            A dictionary of predicted metric values for the next timestep,
            or None if prediction is not possible.
        """
        pass 