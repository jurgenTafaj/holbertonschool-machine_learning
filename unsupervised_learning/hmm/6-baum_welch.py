#!/usr/bin/env python3
"""
module containing function baum_welch
"""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    function that performs the Baum-Welch algorithm for a hidden markov model
    Args:
        Observations: numpy.ndarray of shape (T,)
            that contains the index of the observation
            with T: number of observations
        Transition: numpy.ndarray of shape (M, M)
            that contains the initialized transition probabilities
            with M: number of hidden states
        Emission: numpy.ndarray of shape (M, N)
            that contains the initialized emission probabilities
            with N: number of output states
        Initial: numpy.ndarray of shape (M, 1)
            that contains the initialized starting probabilities
        iterations: number of times expectation-maximization
            should be performed
    Return: the converged Transition, Emission, or None, None on failure
    """
    if type(Observations) != np.ndarray or len(Observations.shape) != 1:
        return None, None
    T = Observations.shape[0]
    if type(T) != int or T < 2:
        return None, None

    M = Transition.shape[0]
    if type(M) != int or M < 1:
        return None, None
    if type(Transition) != np.ndarray or Transition.shape != (M, M):
        return None, None
    if not (np.all(Transition >= 0) and np.all(Transition <= 1)):
        return None, None
    if not np.all(np.sum(Transition, axis=1) == 1):
        return None, None

    N = Emission.shape[1]
    if type(N) != int or N < 1:
        return None, None
    if type(Emission) != np.ndarray or Emission.shape != (M, N):
        return None, None
    if not (np.all(Emission >= 0) and np.all(Emission <= 1)):
        return None, None
    if not np.sum(Emission, axis=1).all():
        return None, None

    if type(Initial) != np.ndarray or Initial.shape != (M, 1):
        return None, None
    if not (np.all(Initial >= 0) and np.all(Initial <= 1)):
        return None, None
    if not np.all(np.sum(Initial) == 1):
        return None, None

    if type(iterations) != int or iterations < 1:
        return None, None

    for _ in range(iterations):
        Transition_prev = np.copy(Transition)
        Emission_prev = np.copy(Emission)
        _, F = forward(Observations, Emission, Transition, Initial)
        _, B = backward(Observations, Emission, Transition, Initial)

        # xsi matrix
        xsi = np.zeros(shape=(M, M, T - 1))
        for t in range(T - 1):
            denominator = np.matmul(np.matmul(
                F[:, t].T, Transition) * Emission[:, Observations[t + 1]].T,
                B[:, t + 1])
            for i in range(M):
                numerator = F[i, t] * Transition[i, :] * Emission[
                    :, Observations[t + 1]].T * B[:, t + 1].T
                with np.errstate(divide='ignore', invalid='ignore'):
                    xsi[i, :, t] = numerator / denominator
        xsi = np.where(~ np.isfinite(xsi), 0., xsi)

        # gamma matrix
        gamma = np.sum(xsi, axis=1)

        # update transition matrix
        with np.errstate(divide='ignore', invalid='ignore'):
            Transition = np.divide(np.sum(xsi, axis=2),
                                   np.sum(gamma, axis=1).reshape(-1, 1))
        Transition = np.where(~ np.isfinite(Transition), 0., Transition)

        # add additional Tth element in gamma
        gamma = np.hstack((gamma, np.sum(xsi[:, :, T - 2], axis=0).reshape(-1,
                                                                           1)))
        gamma = np.where(~ np.isfinite(gamma), 0., gamma)

        # update emission matrix
        denominator = np.sum(gamma, axis=1)
        for output in range(N):
            Emission[:, output] = np.sum(
                gamma[:, Observations == output], axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            Emission = np.divide(Emission,
                                 denominator.reshape(-1, 1))
        Emission = np.where(~ np.isfinite(Emission), 0., Emission)

        if np.isclose(Transition, Transition_prev).all() or \
                np.isclose(Emission, Emission_prev).all():
            return Transition, Emission

    return None, None


def forward(Observation, Emission, Transition, Initial):
    """
    function that performs the forward algorithm for a hidden markov model
    Args:
        Observation:numpy.ndarray of shape (T,)
            that contains the index of the observation
            with T: number of observations
        Emission: numpy.ndarray of shape (N, M)
            containing the emission probability of a specific observation
            given a hidden state
            Emission[i, j]: probability of observing j given the hidden state i
            N: number of hidden states
            M: number of all possible observations
        Transition: 2D numpy.ndarray of shape (N, N)
            containing the transition probabilities
            Transition[i, j]: probability of transitioning from the hidden
            state i to j
        Initial: numpy.ndarray of shape (N, 1)
            containing the probability of starting in a particular hidden state
    Return: P, F, or None, None on failure
        P: likelihood of the observations given the model
        F: numpy.ndarray of shape (N, T)
            containing the forward path probabilities
            F[i, j]: probability of being in hidden state i at time j
            given the previous observations
    """
    if type(Observation) != np.ndarray or len(Observation.shape) != 1:
        return None, None
    T = Observation.shape[0]
    if type(T) != int or T < 1:
        return None, None

    if type(Emission) != np.ndarray or len(Emission.shape) != 2:
        return None, None
    if not (np.all(Emission >= 0) and np.all(Emission <= 1)):
        return None, None
    N = Emission.shape[0]
    M = Emission.shape[1]
    if type(N) != int or N < 1 or type(M) != int or M < 1:
        return None, None

    if type(Transition) != np.ndarray or Transition.shape != (N, N):
        return None, None
    if not (np.all(Transition >= 0) and np.all(Transition <= 1)):
        return None, None

    if type(Initial) != np.ndarray or Initial.shape != (N, 1):
        return None, None
    if not (np.all(Initial >= 0) and np.all(Initial <= 1)):
        return None, None

    F = np.zeros(shape=(N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]

    for t in range(1, T):
        for j in range(N):
            F[j, t] = np.matmul(F[:, t - 1], Transition[:, j]) * Emission[
                j, Observation[t]]

    P = np.sum(F[:, -1])

    return P, F


def backward(Observation, Emission, Transition, Initial):
    """
    function that performs the backward algorithm for a hidden markov model
    Args:
        Observation:numpy.ndarray of shape (T,)
            that contains the index of the observation
            with T: number of observations
        Emission: numpy.ndarray of shape (N, M)
            containing the emission probability of a specific observation
            given a hidden state
            Emission[i, j]: probability of observing j given the hidden state i
            N: number of hidden states
            M: number of all possible observations
        Transition: 2D numpy.ndarray of shape (N, N)
            containing the transition probabilities
            Transition[i, j]: probability of transitioning from the hidden
            state i to j
        Initial: numpy.ndarray of shape (N, 1)
            containing the probability of starting in a particular hidden state
    Return: P, B, or None, None on failure
        P: likelihood of the observations given the model
        B: numpy.ndarray of shape (N, T)
            containing the backward path probabilities
            B[i, j]: probability of generating the future observations from
                hidden state i at time j
    """
    if type(Observation) != np.ndarray or len(Observation.shape) != 1:
        return None, None
    T = Observation.shape[0]
    if type(T) != int or T < 1:
        return None, None

    if type(Emission) != np.ndarray or len(Emission.shape) != 2:
        return None, None
    if not (np.all(Emission >= 0) and np.all(Emission <= 1)):
        return None, None
    N = Emission.shape[0]
    M = Emission.shape[1]
    if type(N) != int or N < 1 or type(M) != int or M < 1:
        return None, None

    if type(Transition) != np.ndarray or Transition.shape != (N, N):
        return None, None
    if not (np.all(Transition >= 0) and np.all(Transition <= 1)):
        return None, None

    if type(Initial) != np.ndarray or Initial.shape != (N, 1):
        return None, None
    if not (np.all(Initial >= 0) and np.all(Initial <= 1)):
        return None, None

    B = np.ones(shape=(N, T))

    for t in range(T - 2, -1, -1):
        for j in range(N):
            B[j, t] = np.sum(B[:, t + 1] * Transition[j, :] *
                             Emission[:, Observation[t + 1]])

    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])

    return P, B
