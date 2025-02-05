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

using namespace TINY;
using namespace tds;

#ifdef USE_TINY
#include "math/tiny/tiny_algebra.hpp"
#else
#include "math/eigen_algebra.hpp"
#endif

int main(int argc, char* argv[]) {
#ifdef USE_TINY
  std::cout << "Using TinyAlgebra" << std::endl;
  typedef TinyAlgebra<double, DoubleUtils> Algebra;
  typedef typename Algebra::Vector3 Vector3;
  typedef typename Algebra::Quaternion Quarternion;
  typedef typename Algebra::VectorX VectorX;
  typedef typename Algebra::Matrix3 Matrix3;
  typedef typename Algebra::Matrix3X Matrix3X;
  typedef typename Algebra::MatrixX MatrixX;
#else
  std::cout << "Using EigenAlgebra" << std::endl;
  typedef EigenAlgebra Algebra;
  typedef typename Algebra::Vector3 Vector3;
  typedef typename Algebra::Quaternion Quarternion;
  typedef typename Algebra::VectorX VectorX;
  typedef typename Algebra::Matrix3 Matrix3;
  typedef typename Algebra::Matrix3X Matrix3X;
  typedef typename Algebra::MatrixX MatrixX;
#endif
  typedef tds::RigidBody<Algebra> RigidBody;
  typedef tds::RigidBodyContactPoint<Algebra> RigidBodyContactPoint;
  typedef tds::MultiBody<Algebra> MultiBody;
  typedef tds::MultiBodyContactPoint<Algebra> MultiBodyContactPoint;
  typedef tds::Transform<Algebra> Transform;

  tds::World<Algebra> world;

  std::vector<RigidBody*> bodies;
  std::vector<int> visuals;

  std::vector<MultiBody*> mbbodies;
  std::vector<int> mbvisuals;

  int num_spheres = 5;

  tds::NeuralNetwork<Algebra> neural_network;
  int observation_size = num_spheres*2;
  int action_size = num_spheres;
  bool use_input_bias = false;
  bool use_output_bias = true;
    
  neural_network.set_input_dim(observation_size, use_input_bias);

  neural_network.add_linear_layer(tds::NN_ACT_IDENTITY, num_spheres, use_output_bias); 

  //  std::vector<Algebra::Scalar> W(observation_size*action_size + use_input_bias*observation_size + use_output_bias*action_size, 0.0);
  std::vector<Algebra::Scalar> W(observation_size*action_size + use_input_bias*observation_size + use_output_bias*action_size, 10.0);
  neural_network.set_parameters(W);

  std::vector<Algebra::Scalar> obs(observation_size, 0.0);
  std::vector<Algebra::Scalar> act(action_size, 0.0);

  MultiBody* mb = world.create_multi_body();
  init_compound_pendulum<Algebra>(*mb, world, num_spheres);

  mbbodies.push_back(mb);

  mb->q() = Algebra::zerox(mb->dof());
  mb->qd() = Algebra::zerox(mb->dof_qd());
  mb->tau() = Algebra::zerox(mb->dof_qd());
  mb->qdd() = Algebra::zerox(mb->dof_qd());

  Vector3 gravity(0., 0., -9.81);

  MatrixX M(mb->links().size(), mb->links().size());

  double dt = 1. / 240.;

  int upAxis = 2;
  for (int i = 0; i < 100; i++) {
    // mb->clear_forces();
    tds::forward_kinematics(*mb);
    
    { world.step(dt); }
    
    for(int i = 0; i < observation_size/2; i++){
      obs[i] = mb->q()[i];
      obs[i+observation_size/2] = mb->qd()[i];
    }
    
    neural_network.compute(obs,act);

    for(int i = 0; i < action_size; i++){
      mb->tau()[i] = act[i];
    }

    // for(int i = 0; i < observation_size; i++){
    //   std::cout << obs[i] << " " ;
    // }
    // std::cout << std::endl;

    // for(int i = 0; i < action_size; i++){
    //   std::cout << act[i] << " " ;
    // }
    
    // std::cout << std::endl;
    
    
    { tds::forward_dynamics(*mb, gravity); }

    {
      tds::integrate_euler(*mb, mb->q(), mb->qd(), mb->qdd(), dt);

      printf("q: [%.3f %.3f] \tqd: [%.3f %.3f]\n", mb->q()[0], mb->q()[1],
             mb->qd()[0], mb->qd()[1]);
      //tds::mass_matrix(*mb, &M);
      //M.print("M");

      if (mb->qd()[0] < -1e4) {
        assert(0);
      }
    }
  }
  return 0;
}
