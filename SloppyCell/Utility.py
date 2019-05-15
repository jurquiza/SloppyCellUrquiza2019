import cPickle
import logging
import smtplib
import sets
from email.MIMEText import MIMEText

import scipy

from scipy import linspace, logspace

import random
import copy

def send_email(to_addr, from_addr=None, subject='', message=''):
    """
    Send a plain-text email to a single address.
    """
    if from_addr is None:
        from_addr = to_addr

    # Create a text/plain message
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['To'], msg['From'] = to_addr, from_addr

    # From the documentation for email... Not sure what it means. :-)
    ## Send the message via our own SMTP server, but don't include the
    ## envelope header.
    s = smtplib.SMTP()
    s.connect()
    s.sendmail(from_addr, [to_addr], msg.as_string())
    s.close()

def save(obj, filename):
    """
    Save an object to a file
    """
    f = file(filename, 'wb')
    cPickle.dump(obj, f, 2)
    f.close()

def load(filename):
    """
    Load an object from a file
    """
    f = file(filename, 'rb')
    obj = cPickle.load(f)
    f.close()
    return obj

def eig(mat):
    """
    Return the sorted eigenvalues and eigenvectors of mat.
    """
    e, v = scipy.linalg.eig(mat)
    order = scipy.argsort(abs(e))[::-1]
    e = scipy.take(e, order)
    v = scipy.take(v, order, axis=1)

    return e, v

def bootstrap(data,num_iterates=100):
    """
    Use sampling with replication to calculate variance in estimates.
    """
    len_data = len(data)
    sampled_data = []
    for ii in range(num_iterates):
        sampled_data.append([random.choice(data) for ii in range(len_data)])

    return sampled_data
        
def enable_debugging_msgs(filename=None):
    """
    Enable output of debugging messages.

    If filename is None messages will be sent to stderr.
    """
    logging.root.setLevel(logging.DEBUG)

    if filename is not None:
        # Remove other handlers
        for h in logging.root.handlers:
            logging.root.removeHandler(h)

        # We need to add a file handler
        handler = logging.FileHandler(filename)
        # For some reason the default file handler format is different.
        # Let's change it back to the default
        formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
        handler.setFormatter(formatter)
        logging.root.addHandler(handler)
    logging.debug('Debug messages enabled.')

def disable_debugging_msgs():
    """
    Disable output of debugging messages.
    """
    logging.root.setLevel(logging.WARN)
    # Remove all other handlers
    for h in logging.root.handlers:
        logging.root.removeHandler(h)
    # Restore basic configuration
    logging.basicConfig()

def disable_warnings():
    scipy.seterr(over='ignore', divide='ignore', invalid='ignore', 
                 under='ignore')
    logging.root.setLevel(logging.CRITICAL)

def enable_warnings():
    logging.root.setLevel(logging.WARNING)
    scipy.seterr(over='print', divide='print', invalid='print', 
                 under='ignore')

class SloppyCellException(Exception):
    pass

class ConstraintViolatedException(SloppyCellException):
    """
    A ConstraintViolatedException is raised when a the value of a network
    constraint becomes False.
    """
    def __init__(self, time, constraint, message):
        self.constraint = constraint
        self.message = message
        self.time = time
    def __str__(self):
        return ('Violated constraint: %s at time %g. Additional info: %s.'
                %(self.constraint, self.time, self.message))

import Redirector_mod
Redirector = Redirector_mod.Redirector

def combine_hessians(hesses, key_sets):
    """
    Combine a number of hessians (with possibly different dimensions and 
    orderings) into a single hessian.

    hesses    A sequence of hessians to combine
    key_sets  A sequence of sequences containing the parameter keys (in order)
              for each hessian.
    """

    # Get a list of all the possible parameter keys
    tot_keys = copy.copy(key_sets[0])
    keys_seen = sets.Set(tot_keys)
    for ks in key_sets[1:]:
        new_keys = [key for key in ks if key not in keys_seen]
        tot_keys.extend(new_keys)
        keys_seen.union_update(sets.Set(new_keys))

    # Add all the appropriate hessian elements together
    key_to_index = dict(zip(tot_keys, range(len(tot_keys))))
    tot_hess = scipy.zeros((len(tot_keys), len(tot_keys)), scipy.float_)
    for hess, keys in zip(hesses, key_sets):
        for ii, id1 in enumerate(keys):
            tot_ii = key_to_index[id1]
            for jj, id2 in enumerate(keys):
                tot_jj = key_to_index[id2]
                tot_hess[tot_ii, tot_jj] += hess[ii, jj]

    return tot_hess, tot_keys

##Functions Introduced by Uriel Urquiza Millar Lab

def SloppyCellData_to_numpy(data, network, chemical):
    """
        Function that retunrs data formated for sloppy cell into numpy arrays for using it within python
        data   Sloppycell dict containing data
        network Name of the data for a particular netowork
        chemical Name of state variables
        returns
        x      time point
        y      chemical
        eps    uncertainty
        """
    
    x =[]
    y = []
    eps=[]
    
    for i in sorted(data[network][chemical].keys()):
        x.append(i)
        y.append(data[network][chemical][i][0])
        eps.append(data[network][chemical][i][1])

    x = np.array(x)
    y = np.array(y)
    eps= np.array(eps)

return x, y , eps

def res_var(res_dict, network, variable):
    
    res_sum = 0
    for res in res_dict.keys():
        if res[1] == network and res[2] == variable:
            res_sum+=res_dict[res]**2
    return res_sum/2

def per_res(res_dict):
    amp_period_res ={}
    for res in res_dict.keys():
        if 'period' in res:
            amp_period_res[res]= res_dict[res]**2/2
    return amp_period_res


def deconv_cost(model, params ,network_data, networks_per_amp_constrais=False):
    deconvoluted_cost = ''
    total_cost=0
    temp = 0
    res_dict = model.res_dict(params)
    deconvoluted_cost+='Network\t'+'Variable\t'+'Chi2\n'
    for n in network_data.keys():
        deconvoluted_cost+=n+'\n'
        for v in network_data[n].keys():
            temp = res_var(res_dict,n,v)
            deconvoluted_cost+= '\t'+v+'\t'+str(temp)+'\n'
            total_cost+=temp
    if networks_per_amp_constrais:
        deconvoluted_cost+='period\n'
        for g in networks_per_amp_constrais:
            for k in per_res(res_dict).keys():
                if g in k and 'per' in k:
                    deconvoluted_cost+='\t'+g+'\t'
                    deconvoluted_cost+=str(networks_per_amp_constrais[g][1]**2*per_res(res_dict)[k]+networks_per_amp_constrais[g][0])
                    deconvoluted_cost+='\t'+str(per_res(res_dict)[k])+'\t'
                    total_cost+=per_res(res_dict)[k]
                    deconvoluted_cost+='\n'
    return deconvoluted_cost + 'Total Chi2\t\t' + str(total_cost)+'\n'

def deconv_cost_to_file(model,params, network_data, networks_per_amp_constrais=False, path='./test.kk'):
    summary = open(path, 'w')
    summary.write(deconv_cost(model,params,network_data, networks_per_amp_constrais))
    summary.close()
    print "File writen to ", path
