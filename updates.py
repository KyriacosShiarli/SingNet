import theano
from theano import tensor as T
from collections import OrderedDict
import numpy as np
import pdb



def adam(params,derivatives,learning_rate,beta1=0.01, beta2=0.001,epsilon=1e-8, gamma=1-1e-8):
    updates = []
    i = theano.shared(np.float32(1))  # HOW to init scalar shared?
    i_t = i + 1.
    fix1 = 1. - (1. - beta1)**i_t
    fix2 = 1. - (1. - beta2)**i_t
    beta1_t = 1-(1-beta1)*gamma**(i_t-1)   # ADDED

    learning_rate_t = learning_rate * (T.sqrt(fix2) / fix1)

    for param_i, g in zip(params, derivatives):
        m = theano.shared(
            np.zeros(param_i.get_value().shape, dtype=theano.config.floatX))
        v = theano.shared(
            np.zeros(param_i.get_value().shape, dtype=theano.config.floatX))

        m_t = (beta1_t * g) + ((1. - beta1_t) * m) # CHANGED from b_t to use beta1_t
        v_t = (beta2 * g**2) + ((1. - beta2) * v)
        g_t = m_t / (T.sqrt(v_t) + epsilon)
        param_i_t = param_i - (learning_rate_t * g_t)

        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((param_i, param_i_t) )
    updates.append((i, i_t))
    return updates
def adadelta(params,derivatives,learning_rate, rho=0.95, epsilon=1e-6):
    """ Adadelta updates
    Scale learning rates by a the ratio of accumulated gradients to accumulated
    step sizes, see [1]_ and notes for further description.
    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps
    rho : float or symbolic scalar
        Squared gradient moving average decay factor
    epsilon : float or symbolic scalar
        Small value added for numerical stability
    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression
    Notes
    -----
    rho should be between 0 and 1. A value of rho close to 1 will decay the
    moving average slowly and a value close to 0 will decay the moving average
    fast.
    rho = 0.95 and epsilon=1e-6 are suggested in the paper and reported to
    work for multiple datasets (MNIST, speech).
    In the paper, no learning rate is considered (so learning_rate=1.0).
    Probably best to keep it at this value.
    epsilon is important for the very first update (so the numerator does
    not become 0).
    Using the step size eta and a decay factor rho the learning rate is
    calculated as:
    .. math::
       r_t &= \\rho r_{t-1} + (1-\\rho)*g^2\\\\
       \\eta_t &= \\eta \\frac{\\sqrt{s_{t-1} + \\epsilon}}
                             {\sqrt{r_t + \epsilon}}\\\\
       s_t &= \\rho s_{t-1} + (1-\\rho)*g^2
    References
    ----------
    .. [1] Zeiler, M. D. (2012):
           ADADELTA: An Adaptive Learning Rate Method.
           arXiv Preprint arXiv:1212.5701.
    """
    updates = OrderedDict()

    for param, grad in zip(params, derivatives):
        value = param.get_value(borrow=True)
        # accu: accumulate gradient magnitudes
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        # delta_accu: accumulate update magnitudes (recursively!)
        delta_accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)

        # update accu (as in rmsprop)
        accu_new = rho * accu + (1 - rho) * grad ** 2
        updates[accu] = accu_new

        # compute parameter update, using the 'old' delta_accu
        update = (grad * T.sqrt(delta_accu + epsilon) /
                  T.sqrt(accu_new + epsilon))
        updates[param] = param - learning_rate * update

        # update delta_accu (as accu, but accumulating updates)
        delta_accu_new = rho * delta_accu + (1 - rho) * update ** 2
        updates[delta_accu] = delta_accu_new

    return updates


def momentum_update(params,derivatives,learning_rate,momentum):
	updates = []
	for param,der in zip(params,derivatives):
		velocity = theano.shared(param.get_value()*0.,broadcastable = param.broadcastable)
		# velocity should be negative
		updates.append((param,param + learning_rate*velocity))
		updates.append((velocity,momentum*velocity - (1.-momentum)*learning_rate*der))
	updates.append((learning_rate,learning_rate*1.000))
	return updates
