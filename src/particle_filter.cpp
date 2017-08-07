/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
  
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 

  num_particles = 50;

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  for (int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;

    particles.push_back(p);
  }
  weights = std::vector<double>(num_particles, 1.0);
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  // default_random_engine gen;

  for(int i = 0; i < num_particles; i++) {
    Particle p = particles[i];

    double x, y, theta;

    // Adding measurements to each particla
    if(fabs(yaw_rate) <= 0.001) {
      // Will lead to divide by zero error. 
      // As yaw_rate is 0, consider linear motion
      x = p.x + velocity * delta_t + cos(p.theta);
      y = p.y + velocity * delta_t + sin(p.theta);
      theta = p.theta;
    } else {
      x = p.x + (velocity/yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
      y = p.y + (velocity/yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
      theta = p.theta + yaw_rate * delta_t;
    }

    // Add random Gaussian noise
    normal_distribution<double> dist_x(x, std_pos[0]);
    normal_distribution<double> dist_y(y, std_pos[1]);
    normal_distribution<double> dist_theta(theta, std_pos[2]);

    // add noise 
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    particles[i] = p;
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations, double sensor_range) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  for(int i = 0; i < observations.size(); i++) {
    double distance = sensor_range;
    int index = -1;
    for (int j = 0; j < predicted.size(); j++) {
      // distance_s dist_factor;
      double d = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if(d < distance) {
        distance = d;
        index = j;
      }
    }
    observations[i].id = index;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  double theta = 0;
  double sigma_x = std_landmark[0];
  double sigma_y = std_landmark[1];
  double sigma_x_sq = 2 * pow(sigma_x, 2); //2*sigma_x * sigma_x;
  double sigma_y_sq = 2 * pow(sigma_y, 2); //2*sigma_y * sigma_y;
  double norm = 1/(2 * M_PI * sigma_x * sigma_y);

  for(int i = 0; i < num_particles; i++) {
    std::vector<LandmarkObs> prediction;
    for(int i = 0; i < map_landmarks.landmark_list.size(); i++) {
      // Filter out landmarks in the map that are beyond the sensor range
      double lx = map_landmarks.landmark_list[i].x_f;
      double ly = map_landmarks.landmark_list[i].y_f;
      int id = map_landmarks.landmark_list[i].id_i;

      if(((particles[i].x - lx)*(particles[i].x - lx) + (particles[i].y - ly) * (particles[i].y - ly)) < sensor_range * sensor_range) {    
        LandmarkObs p_obs;
        p_obs.id = id;
        p_obs.x = lx;
        p_obs.y = ly;
        prediction.push_back(p_obs);
      }
    }

    double total_weight = 0.0;

    // First transform the observations, for current particle
    double x, y;
    x = particles[i].x;
    y = particles[i].y;
    theta = particles[i].theta;

    std::vector<LandmarkObs> T_obs;

    for(int j = 0; j < observations.size(); j++) {
      LandmarkObs obs;
      obs.x = x + observations[j].x * cos(theta) - observations[j].y * sin(theta);
      obs.y = y + observations[j].x * sin(theta) + observations[j].y * cos(theta); 
      T_obs.push_back(obs);
    }

    dataAssociation(prediction, T_obs, sensor_range);

    // Calculate the weights for each observation for this particle
    double weight_p = 1.0;
    for(int j = 0; j < T_obs.size(); j++) {
      int index = T_obs[j].id;
      double diff_x = T_obs[j].x - prediction[index].x;
      double diff_y = T_obs[j].y - prediction[index].y;
      double e = exp(-((pow(diff_x, 2)/ sigma_x_sq) + (pow(diff_y, 2) / sigma_y_sq)));
      weight_p *= norm * e;
    }

    particles[i].weight = weight_p;
    weights[i] = weight_p;
    total_weight += weight_p;
  }
}


void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight. 
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::vector<Particle> temp_p;
  std::vector<double> temp_wt;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> d(weights.begin(), weights.end());
  // std::map<int, int> m;
  for(int n=0; n<num_particles; ++n) {
    int index = d(gen);
    temp_p.push_back(particles[index]);
    temp_wt.push_back(weights[index]);
  }

  particles = temp_p;
  weights = temp_wt;
}


Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}



string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
