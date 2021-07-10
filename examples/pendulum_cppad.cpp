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
const bool use_input_bias = false;
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

    std::vector<AD<double>> parameters(n_params);
    std::vector<double> eval_parameters(n_params);
    double ss = 1;

    // Set NaN trap
    //feenableexcept(FE_INVALID | FE_OVERFLOW);
    
    
    for(int i = 0; i < n_params; i++){
        parameters[i] = 0.01;
        eval_parameters[i] = 0.01;
    }

    for(int i = 0; i < 10000; i ++){
        CppAD::Independent(parameters);
        AD<double> reward;
        
        typedef CppADUtils<double> MyUtils;
        typedef TinyAlgebra<AD<double>, MyUtils> MyAlgebra;
        reward = rollout<MyAlgebra>(parameters);
        
        CppAD::ADFun<double> f(parameters, std::vector<AD<double>>{reward});
        
        vector<double> jac(n_params*1);
        
        jac = f.Jacobian(eval_parameters);

        
        std::cout << i << " " << MyAlgebra::to_double(reward)  << "     " << jac[0] << std::endl;

        
        // for(int j = 0; j < jac.size(); j++){
        //     std::cout << j << " " << jac[j] << std::endl;
        // }

        
        // for(int j = 0; j < jac.size(); j++){
        //   std::cout << j << " " << eval_parameters[j] << std::endl;
        // }
        
        
        for (int j = 0; j < n_params; j++){
            eval_parameters[j] += ss*jac[j];
            parameters[j] = eval_parameters[j];
        }

        // for(int j = 0; j < jac.size(); j++){
        //   std::cout << j << " " << eval_parameters[j] << std::endl;
        // }

    }
    
    
    return 0;
}


// namespace { // begin the empty namespace
//     // define the function Poly(a, x) = a[0] + a[1]*x[1] + ... + a[k-1]*x[k-1]
//     template <class Type>
//     Type Poly(const CPPAD_TESTVECTOR(double) &a, const Type &x)
//     {   size_t k  = a.size();
//         Type y   = 0.;  // initialize summation
//         Type x_i = 1.;  // initialize x^i
//         for(size_t i = 0; i < k; i++)
//         {   y   += a[i] * x_i;  // y   = y + a_i * x^i
//             x_i *= x;           // x_i = x_i * x
//         }
//         return y;
//     }
// }

// int main(void)
// {   using CppAD::AD;   // use AD as abbreviation for CppAD::AD
//     using std::vector; // use vector as abbreviation for std::vector

//     // vector of polynomial coefficients
//     size_t k = 5;                  // number of polynomial coefficients
//     CPPAD_TESTVECTOR(double) a(k); // vector of polynomial coefficients
//     for(size_t i = 0; i < k; i++)
//         a[i] = 1.;                 // value of polynomial coefficients

//     // domain space vector
//     size_t n = 1;               // number of domain space variables
//     vector< AD<double> > ax(n); // vector of domain space variables
//     ax[0] = 3.;                 // value at which function is recorded

//     // declare independent variables and start recording operation sequence
//     CppAD::Independent(ax);

//     // range space vector
//     size_t m = 1;               // number of ranges space variables
//     vector< AD<double> > ay(m); // vector of ranges space variables
//     ay[0] = Poly(a, ax[0]);     // record operations that compute ay[0]

//     // store operation sequence in f: X -> Y and stop recording
//     CppAD::ADFun<double> f(ax, ay);

//     // compute derivative using operation sequence stored in f
//     vector<double> jac(m * n); // Jacobian of f (m by n matrix)
//     vector<double> x(n);       // domain space vector
//     x[0] = 3.;                 // argument value for computing derivative
//     jac  = f.Jacobian(x);      // Jacobian for operation sequence

//     // print the results
//     std::cout << "f'(3) computed by CppAD = " << jac[0] << std::endl;

//     // check if the derivative is correct
//     int error_code;
//     if( jac[0] == 142. )
//         error_code = 0;      // return code for correct case
//     else  error_code = 1;    // return code for incorrect case

//     return error_code;
// }

