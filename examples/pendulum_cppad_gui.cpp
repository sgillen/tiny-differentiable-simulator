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

#include <fenv.h>
#include <chrono>  // std::chrono::seconds
#include <thread>  // std::this_thread::sleep_for



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


#include "visualizer/opengl/tiny_opengl3_app.h"

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
const double torque_limit = 50.0;


TinyOpenGL3App app("pendulum_example_gui", 1024, 768);

template <typename Algebra>
typename Algebra::Scalar rollout_visual(std::vector<typename Algebra::Scalar>  w_policy,
                                  typename Algebra::Scalar dt = Algebra::fraction(1, 60))
{

    using Scalar = typename Algebra::Scalar;
    using Vector3 = typename Algebra::Vector3;

    typedef typename Algebra::Vector3 Vector3;
    typedef typename Algebra::Quaternion Quaternion;
    typedef typename Algebra::VectorX VectorX;
    typedef typename Algebra::Matrix3 Matrix3;
    typedef typename Algebra::Matrix3X Matrix3X;
    typedef typename Algebra::MatrixX MatrixX;
      
    
    typedef tds::RigidBody<Algebra> RigidBody;
    typedef tds::RigidBodyContactPoint<Algebra> RigidBodyContactPoint;
    typedef tds::MultiBody<Algebra> MultiBody;
    typedef tds::MultiBodyContactPoint<Algebra> MultiBodyContactPoint;
    typedef tds::Transform<Algebra> Transform;

    

    
    // typedef tds::RigidBody<Algebra> RigidBody;
    // typedef RigidBody<Algebra> TinyRigidBody;
    // typedef Geometry<Algebra> TinyGeometry;
    // typedef tds::MultiBody<Algebra> MultiBody;
    typedef typename Algebra::MatrixX MatrixX;

    std::vector<RigidBody*> bodies;
    std::vector<int> visuals;
      
    std::vector<MultiBody*> mbbodies;
    std::vector<int> mbvisuals;

    


    tds::NeuralNetwork<Algebra> neural_network;
    neural_network.set_input_dim(observation_size, use_input_bias);
    neural_network.add_linear_layer(tds::NN_ACT_IDENTITY, num_spheres, use_output_bias); 

    
    neural_network.set_parameters(w_policy);
    std::vector<typename Algebra::Scalar> action;

    tds::World<Algebra> world;
    MultiBody* mb = world.create_multi_body();
    mbbodies.push_back(mb);
    init_compound_pendulum<Algebra>(*mb, world, num_spheres);

    int sphere_shape = app.register_graphics_unit_sphere_shape(SPHERE_LOD_HIGH);
    
    
    for (int i = 0; i < num_spheres; i++)
      {
        TinyVector3f pos(0, i*0.1, 0);
        TinyQuaternionf orn(0, 0, 0, 1);
        TinyVector3f color(0.6, 0.6, 1);
        TinyVector3f scaling(0.05, 0.05, 0.05);
        int instance = app.m_renderer->register_graphics_instance(sphere_shape, pos, orn, color, scaling);
        mbvisuals.push_back(instance);
      }
    
    mb->q() = Algebra::zerox(mb->dof());
    mb->qd() = Algebra::zerox(mb->dof_qd());
    mb->tau() = Algebra::zerox(mb->dof_qd());
    mb->qdd() = Algebra::zerox(mb->dof_qd());

    Vector3 gravity(Algebra::zero(), Algebra::zero(), Algebra::from_double(-9.81));

    MatrixX M(mb->links().size(), mb->links().size());

    typename Algebra::Scalar height_vec_sum; 


    app.set_mp4_fps(1./Algebra::to_double(dt));
    int upAxis = 2;

  
    for (int i = 0; i < steps; i++) {
          app.m_renderer->update_camera(upAxis);
          DrawGridData data;
          data.drawAxis = true;
          data.upAxis = upAxis;
          app.draw_grid(data);


          
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
              //mb->tau()[4] = Algebra::from_double(5.0);
            }
          

        }



        { tds::forward_dynamics(*mb, gravity); }
        
        {
            tds::integrate_euler(*mb, mb->q(), mb->qd(), mb->qdd(), dt);
            
            // printf("q: [%.3f %.3f] \tqd: [%.3f %.3f]\n", mb->q()[0], mb->q()[1], mb->qd()[0], mb->qd()[1]);

            double current_h = 0.0;
            for (int j = 0; j < mb->dof(); j++){
              current_h += (Algebra::sin(mb->q()[j])/Algebra::from_double(((double)steps)));
              height_vec_sum += (Algebra::sin(mb->q()[j])/Algebra::from_double(((double)steps)));
            }
            std::cout << "current height: " << current_h << std::endl;
            
            //tds::mass_matrix(*mb, &M);
            //M.print("M");
      
      
            if (mb->qd()[0] < -1e4) {
                assert(0);
            }
        }


        std::this_thread::sleep_for(std::chrono::duration<double>(Algebra::to_double(dt)));
        // sync transforms
        int visual_index = 0;
        TinyVector3f prev_pos(0,0,0);
        TinyVector3f color(0,0,1);
        float line_width = 1;
        
        if (!mbvisuals.empty()) {
          for (int b = 0; b < mbbodies.size(); b++) {
            for (int l = 0; l<mbbodies[b]->links().size();l++) {
              const MultiBody* body = mbbodies[b];
              if (body->links()[l].X_visuals.empty()) continue;
              
              int sphereId = mbvisuals[visual_index++];
              
              Quaternion rot;
              const Transform& geom_X_world = 
                body->links()[l].X_world * body->links()[l].X_visuals[0];
              
              TinyVector3f base_pos(geom_X_world.translation.x(),
                                    geom_X_world.translation.y(),
                                    geom_X_world.translation.z());
              rot = Algebra::matrix_to_quat(geom_X_world.rotation);
              TinyQuaternionf base_orn(rot.x(), rot.y(), rot.z(),
                                       rot.w());
              if (l>=0)
                {
                  //printf("b=%d\n",b);
                  app.m_renderer->draw_line(prev_pos, base_pos,color, line_width);
                }
              else
                {
                  printf("!! b=%d\n",b);
                }
              prev_pos = base_pos;
              app.m_renderer->write_single_instance_transform_to_cpu(base_pos, base_orn, sphereId);
            }
          }
        }
        app.m_renderer->render_scene();
        app.m_renderer->write_transforms();
        app.swap_buffer();
        
        std::this_thread::sleep_for(std::chrono::duration<double>(0.001));
    }
    return height_vec_sum;
};


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
              if (action[i] > Algebra::from_double(torque_limit)){
                action[i] = Algebra::from_double(torque_limit);
              }
              if (action[i] < Algebra::from_double(-torque_limit)){
                action[i] = Algebra::from_double(-torque_limit);
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
{using CppAD::AD;   // use AD as abbreviation for CppAD::AD
  
  app.m_renderer->init();
  app.set_up_axis(2);
  app.m_renderer->get_active_camera()->set_camera_distance(4);
  app.m_renderer->get_active_camera()->set_camera_pitch(-30);
  app.m_renderer->get_active_camera()->set_camera_target_position(0, 0, 0);
  //install ffmpeg in path and uncomment, to enable video recording
  //app.dump_frames_to_video("test.mp4");
  
  std::vector<AD<double>> parameters(n_params);
  std::vector<double> eval_parameters(n_params);


  double ss = 0.02;
  
  for(int i = 0; i < n_params; i++){
    parameters[i] = 0.01;
    eval_parameters[i] = 0.01;
  }
  
  for(int i = 0; i < 100; i ++){
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
      eval_parameters[j] -= ss*jac[j];
      parameters[j] = eval_parameters[j];
    }
    
    if (i % 10 == 0){
      rollout_visual<TinyAlgebra<double, DoubleUtils>>(eval_parameters);
    }
    
    // for(int j = 0; j < jac.size(); j++){
    //   std::cout << j << " " << eval_parameters[j] << std::endl;
    // }
    
  }
  
  
  return 0;
}

