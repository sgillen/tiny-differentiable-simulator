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

const int num_spheres = 5;
const int observation_size = num_spheres*2;
const int action_size = num_spheres;
const bool use_input_bias = false;
const bool use_output_bias = true;
const int n_params = observation_size*action_size + use_input_bias*observation_size + use_output_bias*action_size;
const int steps = 300;

template <typename Algebra>
typename Algebra::Scalar* rollout(const typename Algebra::Scalar* const  w_policy,
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

    std::vector<typename Algebra::Scalar> policy_vec;
    for(int i = 0; i < n_params; i++) {
        policy_vec.push_back(w_policy[i]);
    }


    
    neural_network.set_parameters(policy_vec);
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

    typename Algebra::Scalar* height_vec =  new typename Algebra::Scalar[steps];

  
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
                mb->tau()[i] = action[i];
            }
          

        }



        { tds::forward_dynamics(*mb, gravity); }
        
        {
            tds::integrate_euler(*mb, mb->q(), mb->qd(), mb->qdd(), dt);
            
            // printf("q: [%.3f %.3f] \tqd: [%.3f %.3f]\n", mb->q()[0], mb->q()[1], mb->qd()[0], mb->qd()[1]);

            height_vec[i] = (mb->q()[1]);
            //tds::mass_matrix(*mb, &M);
            //M.print("M");
      
      
            if (mb->qd()[0] < -1e4) {
                assert(0);
            }
        }
    }
  
    return height_vec;
};

// The autodiffed functor, ceres just needs to have the templatized operator() function. 
struct CeresFunctional {
    template <typename T>
    bool operator()(const T* const p, T *e) const{
        //typedef ceres::Jet<double, 2> Jet;
      
        std::cout << "operator" << std::endl;
      
        typedef std::conditional_t<std::is_same_v<T, double>, DoubleUtils, CeresUtils<n_params>> Utils;
        T* tmp = rollout<TinyAlgebra<T, Utils>>(p);
        std::cout << "after calling rollout" << std::endl;
        for(int i = 0; i < steps; i++){
            e[i] = tmp[i];
        }

        std::cout << "returning from functional" << std::endl;
      
        return true;
    }
};



ceres::AutoDiffCostFunction<CeresFunctional, steps, n_params> cost_function(new CeresFunctional);


// int main(int argc, char* argv[]){
//     double* parameters = new double [n_params]();
//     //    double const* const const_paramp = parameters;

//     rollout<TinyAlgebra<double, DoubleUtils>>((double const* const)parameters);
    
    
// }


int main(int argc, char* argv[]) {
    std::cout << "starting" << std::endl;

    double* parameters = new double [n_params];
    double* cost = new double [steps];
    double* gradient = new double[steps*n_params];
    


    for(int i = 0; i < n_params; i++){
        parameters[i] = 0.01;
    }
    
    double const* const* const_paramp = &parameters;

    //    std::cout << cost_function.parameter_block_sizes_.size() << std::endl;
    //    std::cout << steps*n_params<< std::endl;
    
    cost_function.Evaluate(const_paramp, cost, &gradient);
    for(int i = 0; i < steps*n_params; i++){
        std::cout << gradient[i] << std::endl;
    }
    std::cout << "returning from main" << std::endl;
}
