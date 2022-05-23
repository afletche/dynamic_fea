import openmdao.api as om
import numpy as np

class EvaluateDynamicFea(om.ExplicitComponent):

  def initialize(self):
    self.options.declare('fea_object')
    self.options.declare('simp_penalization_factor')
    self.options.declare('filter_radius')
    self.options.declare('loads')
    self.options.declare('t_eval')

  def setup(self):
    fea_object = self.options['fea_object']
    fea_object.setup_dynamics()

    self.add_input('topology_densities', shape=(fea_object.num_elements,))
    self.add_output('weight')
    self.add_output('strain_energy_constraint')
    self.add_output('stress_constraint')
    # self.add_output('U', shape=(fea_object.num_total_dof,))
    self.add_output('dummy_in', value=np.ones((3,3)))
    self.add_output('dummy_out', shape=(3,3))

  def setup_partials(self):
    self.declare_partials('weight', 'topology_densities')
    # self.declare_partials('stress_constraint', 'topology_densities')
    self.declare_partials('strain_energy_constraint', 'topology_densities')
    # self.declare_partials('U', 'topology_densities')
    self.declare_partials('dummy_out', 'dummy_in')


  def compute(self, inputs, outputs):
    fea_object = self.options['fea_object']
    simp_penalization_factor = self.options['simp_penalization_factor']
    filter_radius = self.options['filter_radius']
    loads = self.options['loads']
    t_eval = self.options['t_eval']

    topology_densities = inputs['topology_densities']
    fea_object.evaluate_topology_dynamic(x=topology_densities, simp_penalization_factor=simp_penalization_factor, ramp_penalization_factor=None, filter_radius=filter_radius)
    fea_object.evaluate_dynamics(loads, t_eval, t0=0, x0=None)

    outputs['weight'] =  np.dot(topology_densities, topology_densities)
    # outputs['stress_constraint'] = fea_object.evaluate_stress_constraint(x=topology_densities, eta=1., p=10., epsilon=1.e-2)
    outputs['strain_energy_constraint'] = fea_object.evaluate_strain_energy_constraint()
    print(outputs['strain_energy_constraint'])
    # outputs['U'] = fea_object.U
    outputs['dummy_out'] = fea_object.dummy_out

  def compute_partials(self, inputs, partials, discrete_inputs=None):
    fea_object = self.options['fea_object']
    simp_penalization_factor = self.options['simp_penalization_factor']
    filter_radius = self.options['filter_radius']
    loads = self.options['loads']
    t_eval = self.options['t_eval']

    topology_densities = inputs['topology_densities']

    # fea_object.evaluate_topology_dynamic(x=topology_densities, simp_penalization_factor=simp_penalization_factor, ramp_penalization_factor=None, filter_radius=filter_radius)
    # fea_object.evaluate_dynamics(loads, t_eval, t0=0, x0=None)

    partials['weight', 'topology_densities'] = 2*topology_densities
    # partials['stress_constraint', 'topology_densities'] = fea_object.evaluate_stress_constraint_gradient(x=topology_densities, eta=1., p=10., epsilon=1.e-2)
    partials['strain_energy_constraint', 'topology_densities'] = fea_object.evaluate_strain_energy_gradient(x=topology_densities, simp_penalization_factor=simp_penalization_factor,  ramp_penalization_factor=None, filter_radius=filter_radius, loads=loads, t_eval=t_eval)
    partials['U', 'topology_densities'] = fea_object.evaluate_analytic_test(x=topology_densities)




