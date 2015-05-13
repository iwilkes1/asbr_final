"""
Ian Wilkes
iwilkes1
ianm.wilkes@gmail.com
Algorithms For Sensor Based Robotics Final

Particle Filter
This file contains code implementing a particle filter on the plane.
The only non standard library code it uses is scipy, which runs on all of the
machines in the undergraduate CS laboratory.  It has been hard coded for the
problem in the final to get the predicted x and y positions. 
"""
import numpy
import random
import scipy.stats
import bisect

NUM_PARTICLES = 150000
DEFAULT_ACTION = [0.5, 0]


SENSOR_VARIANCE = 1.0
MOTION_VARIANCE = 0.5
MOTION_STDEV = numpy.sqrt(MOTION_VARIANCE)
SENSOR_STDEV = numpy.sqrt(SENSOR_VARIANCE)

def move_particles(prev_particles, action=DEFAULT_ACTION, 
                   motion_stdev=MOTION_STDEV):
    """
    Uses the movement model to produce new particles based on the previous 
    particles and the specified action.
    """
    new_particles = []
    for particle in prev_particles:
        new_x = numpy.random.normal(particle[0] + action[0], motion_stdev, 1)
        new_y = numpy.random.normal(particle[1] + action[1], motion_stdev, 1)
        new_particles.append((new_x, new_y))
    return new_particles


def get_initial_particles(x_min=0.0, x_max=20.0, y_min=0.0, 
                          y_max=20.0, num_particles=NUM_PARTICLES):
    """
    Initializes the particles based on a uniform distribution over the 
    x,y plane specified. The x and y bounds as well as the number of 
    particles can be specified.
    """
    particles = [];
    for n in range(num_particles):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        particles.append((x,y))
    return particles


def get_weights(particles, sensor_reading=None, sensor_stdev=SENSOR_STDEV):
    """ 
    takes in a set of particles in an iterable, as well as a sensor
    reading, and the sensor standard deviation and produces weights for
    those particles.
    """
    weights = []
    x_read = sensor_reading[0]
    y_read = sensor_reading[1]

    x_dist = scipy.stats.norm(x_read, sensor_stdev)
    y_dist = scipy.stats.norm(y_read, sensor_stdev)
    
    for particle in particles:
        x_prob = x_dist.pdf(particle[0])[0]
        y_prob = y_dist.pdf(particle[1])[0]
        weights.append(x_prob * y_prob)
    return weights

def resample_on_weights(particles, weights):
    """
    Get a new sample of particles based on the weights of the particles
    """
    new_particles = []
    cutoffs = []
    sum = 0.0
    for i in xrange(len(particles)):
        sum += weights[i]
        cutoffs.append(sum)
    for i in xrange(len(particles)):
        search_value = random.uniform(0.0, sum)
        index = bisect.bisect_right(cutoffs, search_value)
        if index:
            new_particles.append(particles[index])
    return new_particles

def get_mean_x(particles):
    """
    calculates the mean x value of the particles
    """
    sum = 0.0;
    for particle in particles:
        sum += particle[0]
    return sum / (len(particles) * 1.0)

def get_mean_y(particles):
    """
    calculates the mean y value of the particles
    """
    sum = 0.0;
    for particle in particles:
        sum += particle[1]
    return sum / (len(particles) * 1.0)


if __name__ == '__main__':
    """ 
    main portion of code.
    """
    x_0 = get_initial_particles()
    print len(x_0), "x_0"
    x_1_prime = move_particles(x_0)
    print len(x_1_prime), "x_1'"
    w_1 = get_weights(x_1_prime, (3.1, 5.3))
    print len(w_1), "w_1'"
    x_1 = resample_on_weights(x_1_prime, w_1)
    print len(x_1), "x_1"
    # no measurements for 2 and 3

    x_2_prime = move_particles(x_1)
    print len(x_2_prime), "x_2'"
    x_3_prime = move_particles(x_2_prime)
    print len(x_3_prime), "x_3'"
    x_4_prime = move_particles(x_3_prime)
    print len(x_4_prime), "x_4'"
    w_4 = get_weights(x_4_prime, (4.7,6.1))
    print len(w_4), "w_4"
    x_4 = resample_on_weights(x_4_prime, w_4)
    print len(x_4), "x_4"
    print get_mean_x(x_4), get_mean_y(x_4)
    


