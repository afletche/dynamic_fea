import openmdao.api as om
import numpy as np

class EvaluateFea(om.ExplicitComponent):

  def initialize(self):
    self.options.declare('fea_object')
    self.options.declare('simp_penalization_factor')
    self.options.declare('filter_radius')

  def setup(self):
    fea_object = self.options['fea_object']
    fea_object.setup()

    self.add_input('topology_densities', shape=(fea_object.num_elements,))
    self.add_output('weight')
    self.add_output('strain_energy_constraint')
    self.add_output('stress_constraint')
    # self.add_output('U', shape=(fea_object.num_free_dof,))
    # self.add_output('U', shape=(1,))

  def setup_partials(self):
    self.declare_partials('weight', 'topology_densities')
    # self.declare_partials('stress_constraint', 'topology_densities')
    self.declare_partials('strain_energy_constraint', 'topology_densities')
    # self.declare_partials('U', 'topology_densities')


  def compute(self, inputs, outputs):
    fea_object = self.options['fea_object']
    simp_penalization_factor = self.options['simp_penalization_factor']
    filter_radius = self.options['filter_radius']

    topology_densities = inputs['topology_densities']
    fea_object.evaluate_topology(x=topology_densities, simp_penalization_factor=simp_penalization_factor, ramp_penalization_factor=None, filter_radius=filter_radius)
    fea_object.evaluate_static()

    # outputs['weight'] =  np.dot(topology_densities, topology_densities)
    outputs['weight'] = np.sum(topology_densities)
    # outputs['stress_constraint'] = fea_object.evaluate_stress_constraint(x=topology_densities, eta=1., p=10., epsilon=1.e-2)
    outputs['strain_energy_constraint'] = fea_object.evaluate_strain_energy_constraint()
    print(outputs['strain_energy_constraint'])
    # outputs['U'] = fea_object.max_strain_energy

  def compute_partials(self, inputs, partials, discrete_inputs=None):
    fea_object = self.options['fea_object']
    simp_penalization_factor = self.options['simp_penalization_factor']
    filter_radius = self.options['filter_radius']

    topology_densities = inputs['topology_densities']

    # fea_object.evaluate_topology(x=topology_densities, simp_penalization_factor=simp_penalization_factor, ramp_penalization_factor=None, filter_radius=filter_radius)
    # fea_object.evaluate_static()

    # partials['weight', 'topology_densities'] = 2*topology_densities
    partials['weight', 'topology_densities'] = np.ones((len(topology_densities),))
    # partials['stress_constraint', 'topology_densities'] = fea_object.evaluate_stress_constraint_gradient(x=topology_densities, eta=1., p=10., epsilon=1.e-2)
    partials['strain_energy_constraint', 'topology_densities'] = fea_object.evaluate_strain_energy_gradient(x=topology_densities, simp_penalization_factor=simp_penalization_factor,  ramp_penalization_factor=None, filter_radius=filter_radius)
    # partials['U', 'topology_densities'] = fea_object.evaluate_analytic_test(x=topology_densities)




