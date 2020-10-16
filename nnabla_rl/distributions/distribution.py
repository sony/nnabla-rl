from abc import ABCMeta, abstractmethod


class Distribution(metaclass=ABCMeta):
    @abstractmethod
    def sample(self, noise_clip=None):
        '''sample

        Sample a value from the distribution

        Args:
            noise_clip(tuple or None): integer tuple of size 2 which contains the minimum and maximum value of the noise to append.

        Returns:
             nnabla.Variable: Sampled value
        '''
        raise NotImplementedError

    def sample_multiple(self, num_samples, noise_clip=None):
        '''sample

        Sample mutiple value from the distribution
        New axis will be added between the first and second axis. 
        Thefore, the returned value shape for mean and variance with shape (batch_size, data_shape)
        will be changed to (batch_size, num_samples, data_shape)

        Args:
            num_samples(int): number of samples per batch
            noise_clip(tuple or None): integer tuple of size 2 which contains the minimum and maximum value of the noise to append.

        Returns:
             nnabla.Variable: Sampled value. 
        '''
        raise NotImplementedError

    def choose_probable(self):
        '''mode

        Compute the most probable action of from the distribution

        Returns:
             nnabla.Variable: Probable action of the distribution

        '''
        raise NotImplementedError

    def mean(self):
        '''mean

        Compute the mean of the distribution (if exist)

        Returns:
             nnabla.Variable: mean of the distribution

        Raises:
             NotImplementedError: The distribution does not have mean
        '''
        raise NotImplementedError

    def log_prob(self, x):
        '''log_prob

        Compute the logarithm of distribution for given input

        Args:
            x (nnabla.Variable): Target value to compute the logarithm of distribution

        Returns: nnabla.Variable
            Logarithm of the probability for given input
        '''
        raise NotImplementedError

    def sample_and_compute_log_prob(self):
        '''log_prob

        Sample a value from the distribution and compute the logarithm of policy distribution

        Args:
            noise_clip(tuple or None): integer tuple of size 2 which contains the minimum and maximum value of the noise to append.

        Returns: (nnabla.Variable, nnabla.Variable)
            Sampled value and its log probabilty
        '''
        raise NotImplementedError

    def entropy(self):
        '''entropy

        Compute the entropy of the distribution

        Returns: (nnabla.Variable)
            Entropy of the distribution
        '''
        raise NotImplementedError

    def kl_divergence(self, q):
        '''kl_divergence

        Compute the kullback leibler divergence between given distribution.
        This function will compute KL(self||target)

        Args:
            q(nnabla_rl.distributions.Distribution): target distribution to compute the kl_divergence

        Returns: (nnabla.Variable)
            Kullback leibler divergence

        '''
        raise NotImplementedError
