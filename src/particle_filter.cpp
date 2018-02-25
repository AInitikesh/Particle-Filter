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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	default_random_engine gen;
	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta
	std_x = std[0];
	std_y = std[1];
	std_theta = std[2];

	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	num_particles = 150;

	for(int i = 0 ; i < num_particles; i++)
	{
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1.0;
		particles.push_back(particle);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;
	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta
	std_x = std_pos[0];
	std_y = std_pos[1];
	std_theta = std_pos[2];

	for(int i = 0 ; i < num_particles; i++)
	{
		double xf, yf, thetaf;
		double x0 = particles[i].x;
		double y0 = particles[i].y;
		double theta0 = particles[i].theta;

		if (yaw_rate == 0) 
		{
			xf = x0 + velocity * delta_t * cos(theta0);
			yf = y0 + velocity * delta_t * sin(theta0);
			thetaf = theta0;
		}
		else{
			xf = x0 + velocity / yaw_rate  * (sin(theta0 + (yaw_rate * delta_t)) - sin(theta0));
			yf = y0 + velocity / yaw_rate  * (cos(theta0) - cos(theta0 + (yaw_rate * delta_t)));
			thetaf = theta0 + yaw_rate * delta_t;
		}

		normal_distribution<double> dist_x(xf, std_x);
		normal_distribution<double> dist_y(yf, std_y);
		normal_distribution<double> dist_theta(thetaf, std_theta);

		particles[i].x =  dist_x(gen);
		particles[i].y =  dist_y(gen);
		particles[i].theta =  dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for(int i = 0; i < observations.size(); i++)
	{
		double min_dist = dist(observations[i].x, observations[i].y, predicted[0].x, predicted[0].y);
		observations[i].id = predicted[0].id;
		for(int j = 1 ; j < predicted.size(); j++)
		{
			LandmarkObs pred = predicted[j];
			double distance = dist(pred.x, pred.y, observations[i].x, observations[i].y);
			if (distance < min_dist)
			{
				min_dist = distance;
				observations[i].id = pred.id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
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
	
	for(int i = 0 ; i < num_particles; i++)
	{
		double x_part= particles[i].x;
		double y_part= particles[i].y;
		double theta= particles[i].theta;
		std::vector<LandmarkObs> trObservations;

		for(int j = 0; j < observations.size(); j++)
		{
			double x_obs= observations[j].x;
			double y_obs= observations[j].y;
			
			LandmarkObs lObs;
			lObs.id = observations[i].id;
			lObs.x = x_part + (cos(theta) * x_obs) - (sin(theta) * y_obs);
			lObs.y = y_part + (sin(theta) * x_obs) + (cos(theta) * y_obs);
			trObservations.push_back(lObs);
		}

		std::vector<LandmarkObs> predicted;

		for(int j = 0 ; j < map_landmarks.landmark_list.size(); j++)
		{
			LandmarkObs lObs;
			lObs.id = map_landmarks.landmark_list[j].id_i;
			lObs.x = map_landmarks.landmark_list[j].x_f;
			lObs.y = map_landmarks.landmark_list[j].y_f;
			double distance = dist(x_part, y_part, lObs.x, lObs.y);
			if (distance <= sensor_range)
			{
				predicted.push_back(lObs);
			}
		}
		dataAssociation(predicted,trObservations);

		double sig_x = std_landmark[0];
		double sig_y = std_landmark[1];

		// calculate normalization term
		double gauss_norm = (1.0/(2.0 * M_PI * sig_x * sig_y));

		particles[i].weight = 1.0;

		for(int j = 0; j < trObservations.size(); j++)
		{
			LandmarkObs currObs = trObservations[j]; 
			double x_obs= currObs.x;
			double y_obs= currObs.y;

			double mu_x = 0;
			double mu_y = 0.0;
			for(int k = 0 ; k < predicted.size(); k++)
			{
				LandmarkObs currPred = predicted[k];
				if(currObs.id == currPred.id)
				{
					mu_x = currPred.x;
					mu_y = currPred.y;
				}
			}

			// calculate exponent
			double exponent = ((pow((x_obs - mu_x),2))/(2 * pow(sig_x,2))) + ((pow((y_obs - mu_y),2))/(2 * pow(sig_y,2)));
			double weight = gauss_norm * exp(-exponent);

			// calculate weight using normalization terms and exponent
			particles[i].weight *= weight;
		}
		
	}


}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<Particle> new_particles;

	// get all of the current weights
	vector<double> weights;
	for (int i = 0; i < num_particles; i++) {
		weights.push_back(particles[i].weight);
	}
	default_random_engine gen;
    std::discrete_distribution<int> d(weights.begin(), weights.end());
    
    for(int i=0; i<num_particles; i++) {
        new_particles.push_back(particles[d(gen)]);
    }
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
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