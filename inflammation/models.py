"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains 
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.
"""

import numpy as np


# %%
class Observation:
    """Class for storing day-value pairs representing inflammation observations.
    """
    def __init__(self, day, value):
        self.day = day
        self.value = value
    
    def __str__(self):
        return str(self.value)
    

# %%
class Person:
    """Class defining a person object with a name. 
    """
    def __init__(self, name: str):
        self.name = name
    
    def __str__(self):
        return self.name


# %%
class Doctor(Person):
    """_summary_

    Args:
        Person (_type_): _description_
    """
    def __init__(self, name: str):
        super().__init__(name)
        self.patients = {}
    
    @property
    def patient_names(self):
        return [patient.name for patient in patients]
    
    def add_patient(self, patient):
        self.patients[patient.name] = patient


# %%
class Patient(Person):
    """_summary_

    Args:
        Person (_type_): _description_
    """
    def __init__(self, name: str):
        super().__init__(name)
        self.observations = []
    
    @property
    def last_observation(self): return self.observations[-1]

    def add_observation(self, value, day = None):
        if day is None:
            try:
                day = self.observations[-1].day + 1
            except IndexError:
                day = 0
        new_observation = Observation(day, value)
        self.observations.append(new_observation)
        return new_observation


def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname=filename, delimiter=',')


def daily_mean(data):
    """Calculate the daily mean of a 2d inflammation data array."""
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2d inflammation data array."""
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2d inflammation data array."""
    return np.min(data, axis=0)


def daily_std(data):
    """Calculate the daily standard deviation of a 2d inflammation data array."""
    return np.std(data, axis = 0)


def patient_normalise(data):
    """Normalise patient data from 2D array of inflammation data"""
    if np.any(data < 0): 
        raise ValueError("Inflammation values should not be negative")
    maxes = np.nanmax(data, axis = 1)
    with np.errstate(invalid = "ignore", divide = "ignore"):
        normalised = data/maxes[:, np.newaxis]
    normalised[np.isnan(normalised)] = 0
    normalised[normalised < 0] = 0
    return normalised