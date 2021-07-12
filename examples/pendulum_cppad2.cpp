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




#include <iostream>        // standard input/output
#include <vector>          // standard vector
#include "math/tiny/cppad_utils.h"
#include <cppad/cppad.hpp> // the CppAD package

using namespace TINY;
using namespace tds;

#include "math/tiny/tiny_algebra.hpp"

const int num_spheres = 1;
const int observation_size = num_spheres*2;
const int action_size = num_spheres;
const bool use_input_bias = true;
const bool use_output_bias = true;
const int n_params = observation_size*action_size + use_input_bias*observation_size + use_output_bias*action_size;
const int steps = 300;

template <typename Algebra>
typename Algebra::Scalar rollout(std::vector<typename Algebra::Scalar>  w_policy,
                                  typename Algebra::Scalar dt = Algebra::fraction(1, 60))
{

    using Scalar = typename Algebra::Scalar;
    using Vector3 = typename Algebra::Vector3;
    
    typedef RigidBody<Algebra> TinyRigidBody;
    typedef Geometry<Algebra> TinyGeometry;
    typedef tds::MultiBody<Algebra> MultiBody;
    typedef typename Algebra::MatrixX MatrixX;

    tds::NeuralNetwork<Algebra> neural_network;
    neural_network.set_input_dim(observation_size, use_input_bias);
    neural_network.add_linear_layer(tds::NN_ACT_IDENTITY, num_spheres, use_output_bias); 

    
    neural_network.set_parameters(w_policy);
    std::vector<typename Algebra::Scalar> action;

    tds::World<Algebra> world;
    MultiBody* mb = world.create_multi_body();
  
    init_compound_pendulum<Algebra>(*mb, world, num_spheres);

    mb->q() = Algebra::zerox(mb->dof());
    mb->qd() = Algebra::zerox(mb->dof_qd());
    mb->tau() = Algebra::zerox(mb->dof_qd());
    mb->qdd() = Algebra::zerox(mb->dof_qd());

    Vector3 gravity(Algebra::zero(), Algebra::zero(), Algebra::from_double(-9.81));

    MatrixX M(mb->links().size(), mb->links().size());

    typename Algebra::Scalar height_vec_sum; 

  
    for (int i = 0; i < steps; i++) {
        tds::forward_kinematics(*mb);            

        { world.step(dt); }

                
        {
                            
            std::vector<typename Algebra::Scalar> observation(observation_size);
            std::vector<typename Algebra::Scalar> action(action_size);
            //          typename MultiBody::VectorX tau(action_size);
          
            for(int i = 0; i<mb->dof(); i++){
                observation[i] = mb->q()[i];
            }

            for(int i = 0; i<mb->dof_qd(); i++){
                observation[i+mb->dof()] = mb->qd()[i];
            }

            neural_network.compute(observation, action);

            for(int i = 0; i<action_size; i++){
              if (action[i] > Algebra::from_double(5.0)){
                action[i] = Algebra::from_double(5.0);
              }
              if (action[i] < Algebra::from_double(-5.0)){
                action[i] = Algebra::from_double(-5.0);
              }

              mb->tau()[i] = action[i];
            }
          

        }



        { tds::forward_dynamics(*mb, gravity); }
        
        {
            tds::integrate_euler(*mb, mb->q(), mb->qd(), mb->qdd(), dt);
            
            // printf("q: [%.3f %.3f] \tqd: [%.3f %.3f]\n", mb->q()[0], mb->q()[1], mb->qd()[0], mb->qd()[1]);

            for (int j = 0; j < mb->dof(); j++){
              height_vec_sum += (Algebra::sin(mb->q()[j])/Algebra::from_double(((double)steps)));
            }
            
            //tds::mass_matrix(*mb, &M);
            //M.print("M");
      
      
            if (mb->qd()[0] < -1e4) {
                assert(0);
            }
        }
    }

   
    
    return height_vec_sum;
};



int main(void)
{   using CppAD::AD;   // use AD as abbreviation for CppAD::AD

    std::vector<AD<double>> ad_params(n_params);
    std::vector<double> db_params(n_params);
    double ss = .01;

    // Set NaN trap
    //feenableexcept(FE_INVALID | FE_OVERFLOW);
    
    
    for(int i = 0; i < n_params; i++){
        db_params[i] = 0.01;
        ad_params[i] = 0.01;
    }

    typedef CppADUtils<double> MyUtils;
    typedef TinyAlgebra<AD<double>, MyUtils> ADAlgebra;
    typedef TinyAlgebra<double, DoubleUtils> DoubleAlgebra;
    
    AD<double> ad_reward;
    double db_reward;
    vector<double> jac(n_params*1);

      

    
    for(int i = 0; i < 100; i ++){
      CppAD::Independent(ad_params);
      ad_reward = rollout<ADAlgebra>(ad_params);
      CppAD::ADFun<double> f(ad_params, std::vector<AD<double>>{ad_reward});
      
      
      db_reward = rollout<DoubleAlgebra>(db_params);
      jac = f.Jacobian(db_params);
      
      
        
      std::cout << i << " " << db_reward  << "     " << jac[0] << std::endl;
      
      // for(int j = 0; j < jac.size(); j++){
      //     std::cout << j << " " << jac[j] << std::endl;
      // }
      
      
      // for(int j = 0; j < jac.size(); j++){
      //   std::cout << j << " " << eval_parameters[j] << std::endl;
      // }
      
      
      for (int j = 0; j < n_params; j++){
	db_params[j] += ss*jac[j];
	ad_params[j] = db_params[j];
      }

        // for(int j = 0; j < jac.size(); j++){
        //   std::cout << j << " " << eval_parameters[j] << std::endl;
        // }

    }
    
    
    return 0;
}

