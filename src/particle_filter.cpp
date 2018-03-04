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

void ParticleFilter::init(double x, double y, double theta, double std[])
{
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = 10;

    particles.resize(num_particles);
    weights.resize(num_particles);

    default_random_engine gen;
    double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

    // Set standard deviations for x, y, and theta
    std_x = std[0];
    std_y = std[1];
    std_theta = std[2];

    // These lines creates a normal (Gaussian) distribution for x, y and theta
    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    for (int i = 0; i < num_particles; ++i)
    {
        Particle p;
        p.weight = 1;

        // Sample and from these normal distributions like this:
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);

        // Add particle to the vector
        particles[i] = p;
        weights[i] = 1;
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine gen;
    double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

    // Set standard deviations for x, y, and theta
    std_x = std_pos[0];
    std_y = std_pos[1];
    std_theta = std_pos[2];

    for (int i = 0; i < num_particles; ++i)
    {
        double x = particles[i].x;
        double y = particles[i].y;
        double theta = particles[i].theta;

        if (fabs(yaw_rate) < 0.00001)
        {
            x = x + velocity * delta_t * cos(theta);
            y = y + velocity * delta_t * sin(theta);
        }
        else
        {
            x = x + (velocity / yaw_rate) * (sin(theta + yaw_rate * delta_t) - sin(theta));
            y = y + (velocity / yaw_rate) * (cos(theta) - cos(theta + yaw_rate * delta_t));
            theta = theta + yaw_rate * delta_t;
        }

        // These lines creates a normal (Gaussian) distribution for x, y and theta
        normal_distribution<double> dist_x(x, std_x);
        normal_distribution<double> dist_y(y, std_y);
        normal_distribution<double> dist_theta(theta, std_theta);

        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map)
{
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

    double sig_x = std_landmark[0];
    double sig_y = std_landmark[1];

    for (int i = 0; i < num_particles; ++i)
    {
        particles[i].weight = 1.0f;

        std::vector<double> sense_x;
        std::vector<double> sense_y;
        std::vector<int> associations;

        for (int j = 0; j < observations.size(); ++j)
        {
            Particle p = particles[i];

            // Convert observation from VEHICLE's coordinate system to MAP's
            LandmarkObs observation;
            observation.x = p.x + (cos(p.theta) * observations[j].x - sin(p.theta) * observations[j].y);
            observation.y = p.y + (sin(p.theta) * observations[j].x + cos(p.theta) * observations[j].y);

            sense_x.push_back(observation.x);
            sense_y.push_back(observation.y);

            // Get all landmarks based on sensor range
            vector<Map::single_landmark_s> landmarks = GetLandmarksBySensorRange(sensor_range, p.x, p.y, map);

            // Get nearest landmark for current observation
            Map::single_landmark_s nearest_landmark = GetNearestLandmark(observation, landmarks);
            associations.push_back(nearest_landmark.id_i);

            particles[i] = SetAssociations(particles[i], associations, sense_x, sense_y);

            // Calculate weight
            float mu_x = nearest_landmark.x_f;
            float mu_y = nearest_landmark.y_f;
            float diffX = observation.x - mu_x;
            float diffY = observation.y - mu_y;
            double exponent = ((diffX * diffX) / (2 * sig_x * sig_x)) + ((diffY * diffY) / (2 * sig_y * sig_y));
            float weight = (1.0f / (2 * M_PI * sig_x * sig_y)) * exp(-exponent);

            particles[i].weight *= weight;
        }

        weights[i] = particles[i].weight;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    default_random_engine gen;
    discrete_distribution<int> distribution(weights.begin(), weights.end());

    vector<Particle> resampledParticles;
    for (int i = 0; i < num_particles; ++i)
    {
        int index = distribution(gen);
        resampledParticles.push_back(particles[index]);
    }

    particles = resampledParticles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    // particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

vector<Map::single_landmark_s> ParticleFilter::GetLandmarksBySensorRange(double sensor_range, double x, double y, const Map &map)
{
    std::vector<Map::single_landmark_s> landmarks;

    for (int i = 0; i < map.landmark_list.size(); ++i)
    {
        if (fabs(map.landmark_list[i].x_f - x) <= sensor_range && fabs(map.landmark_list[i].y_f - y) <= sensor_range)
        {
            landmarks.push_back(map.landmark_list[i]);
        }
    }

    return landmarks;
}

Map::single_landmark_s ParticleFilter::GetNearestLandmark(LandmarkObs observation, vector<Map::single_landmark_s> landmarks)
{
    Map::single_landmark_s nearest_landmark;
    double min_dist = std::numeric_limits<double>::max();

    for (int l = 0; l < landmarks.size(); ++l)
    {
        double distance = dist(observation.x, observation.y, landmarks[l].x_f, landmarks[l].y_f);

        if (distance < min_dist)
        {
            nearest_landmark.id_i = landmarks[l].id_i;
            nearest_landmark.x_f = landmarks[l].x_f;
            nearest_landmark.y_f = landmarks[l].y_f;
            min_dist = distance;
        }
    }

    return nearest_landmark;
}