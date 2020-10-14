from abc import ABC, abstractmethod


class ModelBase(ABC):
    """Common interface for models to be used in sensitivity analysis problems.

    This is an abstract class, you must inherit from it instead of using it directly. Subclasses are **required** to define the three abstract methods given below."""

    @abstractmethod
    def __len__(self):
        """Must return number of model parameters"""
        pass

    @abstractmethod
    def rescale(self, unit_interval_matrix):
        """Must rescale an 2-d array from [0-1] space to parameter space.

        ``unit_interval_matrix`` has parameters as rows, and Monte Carlo sampling iterations as columns."""
        pass

    @abstractmethod
    def __call__(self, input_parameter_values):
        """Execute the model with the input values in ``input_parameter_values``.

        ``input_parameter_values`` has parameters as rows, and Monte Carlo sampling iterations as columns.

        Must return a 1-d vector with same length as the number of rows in ``input_parameter_values``."""
        pass
