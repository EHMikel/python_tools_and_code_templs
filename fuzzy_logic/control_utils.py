import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from skfuzzy.control.controlsystem import CrispValueCalculator


def plot_universe(variable, figsize=(14, 6), title:str=None , xlabel:str=None, ylabel:str='Grado de Pertenencia', legend:bool=True):
    """
    Función para graficar los conjuntos difusos de una variable en Scikit-Fuzzy.
    """
    plt.figure(figsize=figsize)

    # Graficar cada conjunto difuso de la variable
    for term in variable.terms:
        plt.plot(variable.universe, variable[term].mf, label=term)

    if title == None: title = variable.label        # Añadir título y etiquetas si se proporcionan
    if xlabel: plt.xlabel(xlabel)                   # mostrar etiqutea del eje x
    if legend: plt.legend(loc= 'best')              # Mostrar leyenda si se indica
    
    plt.title(title, loc= 'center', fontsize= 18)
    plt.ylabel(ylabel)
    # Mostrar el gráfico
    plt.show()


def plot_fuzzy_simulation(variable, simulation, figsize=(14, 6), title=None, xlabel=None, ylabel='Grado de Pertenencia'):
    """
    Visualiza la variable difusa y sus funciones de membresía dada una simulación.
    """
    if simulation is None:  simulation = ctrl.ControlSystemSimulation(ctrl.ControlSystem()) # Crear una simulación vacía para visualizar con valores predeterminados

    fig, ax = plt.subplots(figsize=figsize)

    crispy = CrispValueCalculator(variable, simulation)
    ups_universe, output_mf, cut_mfs = crispy.find_memberships()
    zeros = np.zeros_like(ups_universe, dtype=np.float64)

    for label, mf in variable.terms.items():
        ax.plot(variable.universe, mf.mf, label=label)   # ax.plot(ups_universe, mf.mf, label=label)
        if label in cut_mfs:    ax.fill_between(ups_universe, zeros, cut_mfs[label], alpha=0.4)
    
    input_vars = simulation._get_inputs()
    if variable.label in input_vars.keys(): mi_varible = simulation._get_inputs()[variable.label]
    else:                                   mi_varible = simulation.output[variable.label]
    
    plt.axvline(x= mi_varible)

    # Personalizar el gráfico
    if title == None:  ax.set_title(variable.label, fontsize= 18)
    else:              ax.set_title(title, fontsize= 18)
    if xlabel:         ax.set_xlabel(xlabel)
        
    ax.set_ylabel(ylabel)
    ax.legend(loc= 'best')

    plt.show()