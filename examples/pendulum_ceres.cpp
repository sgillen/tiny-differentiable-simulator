// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>

#include <iostream>  // std::cout

#include "dynamics/forward_dynamics.hpp"
#include "dynamics/integrator.hpp"
#include "dynamics/kinematics.hpp"
#include "math/tiny/fix64_scalar.h"
#include "math/tiny/tiny_double_utils.h"
#include "multi_body.hpp"
#include "utils/file_utils.hpp"
#include "utils/pendulum.hpp"
#include "world.hpp"
#include "math/neural_network.hpp"



#include "math/tiny/ceres_utils.h"
#include <ceres/autodiff_cost_function.h>
#include "math/tiny/ceres_utils.h"


using namespace TINY;
using namespace tds;

#include "math/tiny/tiny_algebra.hpp"

template <typename Algebra>
std::vector<typename Algebra::Scalar> rollout(const std::vector<typename Algebra::Scalar>& w_policy, int steps = 100, typename Algebra::Scalar dt = Algebra::fraction(1, 60)) {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  
  typedef RigidBody<Algebra> TinyRigidBody;
  typedef Geometry<Algebra> TinyGeometry;
  typedef tds::MultiBody<Algebra> MultiBody;
  typedef typename Algebra::MatrixX MatrixX;


  tds::NeuralNetwork<Algebra> neural_network;
  neural_network.set_parameters(w_policy);
  std::vector<typename Algebra::Scalar> action;
      
  


  tds::World<Algebra> world;
  MultiBody* mb = world.create_multi_body();
  
  int num_spheres = 2;
  init_compound_pendulum<Algebra>(*mb, world, num_spheres);

  mb->q() = Algebra::zerox(mb->dof());
  mb->qd() = Algebra::zerox(mb->dof_qd());
  mb->tau() = Algebra::zerox(mb->dof_qd());
  mb->qdd() = Algebra::zerox(mb->dof_qd());

  Vector3 gravity(0., 0., -9.81);

  MatrixX M(mb->links().size(), mb->links().size());

  std::vector<typename Algebra::Scalar> hvec;
  hvec.reserve(steps);
  
  for (int i = 0; i < steps; i++) {
    // mb->clear_forces();
    tds::forward_kinematics(*mb);

    { world.step(dt); }
    
    neural_network.compute(mb->q(), action);
    mb->tau[0] = action[0];
    mb->tau[1] = action[1];
    

    { tds::forward_dynamics(*mb, gravity); }
    
    {
      tds::integrate_euler(*mb, mb->q(), mb->qd(), mb->qdd(), dt);
      
      printf("q: [%.3f %.3f] \tqd: [%.3f %.3f]\n", mb->q()[0], mb->q()[1],
             mb->qd()[0], mb->qd()[1]);

      hvec.push_back(Algebra::sin1(mb->q()[1]));
      //tds::mass_matrix(*mb, &M);
      //M.print("M");
      
      
      if (mb->qd()[0] < -1e4) {
        assert(0);
      }
    }
  }
  
  return hvec;
};



struct CeresFunctional {
  int steps{100};
  
  template <typename T>
  bool operator()(const T* const p, T *e) const{
    //typedef ceres::Jet<double, 2> Jet;
    std::vector<T> params{p[0], p[1], p[2], p[3]};
    typedef std::conditional_t<std::is_same_v<T, double>, DoubleUtils, CeresUtils<2>> Utils;
    *e = rollout<TinyAlgebra<T, Utils>>(params);
    return true;
  }
};

ceres::AutoDiffCostFunction<CeresFunctional, 1, 4> cost_function(new CeresFunctional);
double* parameters = new double[4];
double* gradient = new double[4];


void grad_ceres(double* cost, int steps = 300) {
  double const* const* params = &parameters;
  cost_function.Evaluate(params, cost, &gradient);
}



int main(int argc, char* argv[]) {
  
  
}
