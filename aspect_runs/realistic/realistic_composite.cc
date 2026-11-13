/*
  Copyright (C) 2026 by the authors of the MantleSEM project.

  This file is part of a plugin for ASPECT, distributed under the same
  license as ASPECT (GNU GPL v2 or later).

  ---------------------------------------------------------------------------

  `realistic composite`: ASPECT's built-in `diffusion dislocation` material
  model, extended with the two solid-solid phase transitions of the
  "realistic 2-D case" (z0 = 0.87 and 0.78, both with a tanh width of 0.025)
  and their latent heat.

  The rheology is *literally* ASPECT's `DiffusionDislocation<dim>::evaluate`
  (the class is used as the base class and its evaluate() is called first), so
  the viscosity is the isostress partition of the diffusion and dislocation
  creep laws with ASPECT's Newton iteration and clamp.  This plugin only adds

    * the phase-function density jumps  rho += sum_k Gamma_k * Delta rho_k
      (ASPECT's `latent heat` material model convention, positive = denser
      with depth), and

    * the entropy derivatives  dS/dp = sum_k Gamma_p,k * Delta S_k,
      dS/dT = -sum_k Gamma_p,k * Delta S_k * gamma_k  with
      Delta S_k = gamma_k Delta rho_k / rho^2, which the built-in
      `latent heat` heating model turns into the latent-heat source and the
      apparent heat capacity (identical formulas to
      source/material_model/latent_heat.cc of ASPECT 2.5.0-pre).

  The phase function itself is ASPECT's own MaterialUtilities::PhaseFunction,
  so the phase value/derivative conventions are the library ones (depth based,
  Clapeyron slope in Pa/K).

  The thermal conductivity is forced to kappa * c_p * rho_ref (constant), which
  is the constant-conductivity model of the MantleSEM case; the built-in model
  would use kappa * c_p * rho(T) and thus a temperature-dependent k.
*/

#include <aspect/material_model/diffusion_dislocation.h>
#include <aspect/material_model/utilities.h>
#include <aspect/adiabatic_conditions/interface.h>
#include <aspect/geometry_model/interface.h>
#include <aspect/gravity_model/interface.h>

#include <deal.II/base/parameter_handler.h>

namespace aspect
{
  namespace MaterialModel
  {
    using namespace dealii;

    template <int dim>
    class RealisticComposite : public DiffusionDislocation<dim>
    {
      public:
        void
        evaluate (const MaterialModel::MaterialModelInputs<dim> &in,
                  MaterialModel::MaterialModelOutputs<dim> &out) const override;

        static
        void
        declare_parameters (ParameterHandler &prm);

        void
        parse_parameters (ParameterHandler &prm) override;

      private:
        MaterialUtilities::PhaseFunction<dim> phase_function;

        std::vector<double> density_jumps;

        double reference_density;
        double thermal_diffusivity;
        double heat_capacity;
    };



    template <int dim>
    void
    RealisticComposite<dim>::
    evaluate (const MaterialModel::MaterialModelInputs<dim> &in,
              MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      // The rheology, the density (without phases) and all other outputs are
      // exactly those of the built-in model.
      DiffusionDislocation<dim>::evaluate (in, out);

      const bool include_latent = this->include_latent_heat();

      // Phase-function inputs that do not depend on the evaluation point.
      const unsigned int n_transitions = phase_function.n_phase_transitions();

      for (unsigned int i=0; i<in.n_evaluation_points(); ++i)
        {
          // Constant conductivity of the MantleSEM case (the built-in model
          // uses kappa*c_p*rho(T)).
          out.thermal_conductivities[i] = thermal_diffusivity * heat_capacity * reference_density;

          const double depth = this->get_geometry_model().depth(in.position[i]);
          const double pressure_depth_derivative
            = (depth > 0.0
               ?
               this->get_adiabatic_conditions().pressure(in.position[i]) / depth
               :
               this->get_gravity_model().gravity_vector(in.position[i]).norm() * reference_density);

          std::vector<double> phase_values (n_transitions);
          double phase_dependence = 0.0;

          for (unsigned int phase=0; phase<n_transitions; ++phase)
            {
              const MaterialUtilities::PhaseFunctionInputs<dim> phase_in (in.temperature[i],
                                                                          in.pressure[i],
                                                                          depth,
                                                                          pressure_depth_derivative,
                                                                          phase);
              phase_values[phase] = phase_function.compute_value(phase_in);
              phase_dependence += phase_values[phase] * density_jumps[phase];
            }

          // Boussinesq: the phase jump is added to the linearized density
          // exactly as in ASPECT's `latent heat` material model.
          out.densities[i] += phase_dependence;

          if (include_latent)
            {
              const double rho = out.densities[i];
              double entropy_gradient_pressure    = 0.0;
              double entropy_gradient_temperature = 0.0;

              for (unsigned int phase=0; phase<n_transitions; ++phase)
                {
                  const MaterialUtilities::PhaseFunctionInputs<dim> phase_in (in.temperature[i],
                                                                              in.pressure[i],
                                                                              depth,
                                                                              pressure_depth_derivative,
                                                                              phase);
                  const double phase_function_derivative = phase_function.compute_derivative(phase_in);
                  const double clapeyron_slope = phase_function.get_transition_slope(phase);

                  // ASPECT's definition: dS = gamma * drho / rho^2 (this is
                  // minus the physical entropy change, which is what makes the
                  // latent-heat source in ASPECT heat a downwelling at a
                  // positive-Clapeyron transition).
                  const double entropy_change = clapeyron_slope * density_jumps[phase] / (rho * rho);

                  entropy_gradient_pressure    += phase_function_derivative * entropy_change;
                  entropy_gradient_temperature -= phase_function_derivative * entropy_change * clapeyron_slope;
                }

              out.entropy_derivative_pressure[i]    = entropy_gradient_pressure;
              out.entropy_derivative_temperature[i] = entropy_gradient_temperature;
            }
        }
    }



    template <int dim>
    void
    RealisticComposite<dim>::declare_parameters (ParameterHandler &prm)
    {
      DiffusionDislocation<dim>::declare_parameters (prm);

      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Diffusion dislocation");
        {
          MaterialUtilities::PhaseFunction<dim>::declare_parameters (prm);

          prm.declare_entry ("Phase transition density jumps", "",
                             Patterns::List(Patterns::Double(0.)),
                             "A list of density jumps at each phase transition. A positive value means "
                             "that the density increases with depth. Units: kg/m^3.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    RealisticComposite<dim>::parse_parameters (ParameterHandler &prm)
    {
      DiffusionDislocation<dim>::parse_parameters (prm);

      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Diffusion dislocation");
        {
          phase_function.initialize_simulator (this->get_simulator());
          phase_function.parse_parameters (prm);

          density_jumps = Utilities::string_to_double
                          (Utilities::split_string_list(prm.get("Phase transition density jumps")));

          // Background density and the two constants the built-in model uses
          // internally; re-read here because they are private in the base.
          reference_density   = Utilities::string_to_double
                                (Utilities::split_string_list(prm.get("Densities")))[0];
          thermal_diffusivity = prm.get_double("Thermal diffusivity");
          heat_capacity       = prm.get_double("Heat capacity");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(RealisticComposite,
                                   "realistic composite",
                                   "ASPECT's `diffusion dislocation` rheology plus the two "
                                   "phase transitions (density jumps and latent heat) of the "
                                   "MantleSEM realistic 2-D case. The phase parameters are the "
                                   "ones of ASPECT's `latent heat` material model.")
  }
}
