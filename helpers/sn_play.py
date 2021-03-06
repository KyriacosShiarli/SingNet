from preprocess import pickle_loader,map_to_range_symmetric,map_to_range
#import pygame
import numpy as np
import theano

import pdb
import matplotlib.pyplot as plt
from sn_plot import plot_filters
from preprocess import pickle_loader
import scipy.io.wavfile as wav


def device_play(nnet,sample_rate,duration = 100,buffer_size=20):
    buffer_size = 20 # sample buffer for playback. Hevent really determined what i does qualitatively. Probably affects latency
    play_duration = duration # play duration in miliseconds
    pygame.mixer.pre_init(sample_rate, -16, 2,buffer = buffer_size) # 44.1kHz, 16-bit signed, stereo
    pygame.init()
    volLeft = 0.5;volRight=0.5 # Volume
    z_val = np.zeros(nnet.dim_z) # Initial latent variable values
    port = mido.open_input(mido.get_input_names()[0]) # midi port. chooses the first midi device it detects.
    device_range = [0,127.]
    wanted_range = [-10.,10.]
    while True:
        mu_out = nnet.generate(z_val)
        for msg in port.iter_pending():
            if msg.channel < nnet.dim_z:
                print msg.channel
                print "+++++++++++++++++++++++"
                z_val[msg.channel] = map_to_range(msg.value,device_range,wanted_range)
            else:
                print "Midi channel beyond latent variables"
        mu_out = map_to_range_symmetric(mu_out,[-1,1],[-32768,32768],from_data=True)
        mu_out = mu_out.astype(np.int16)
        mu_out = np.array(zip(mu_out,mu_out)) # make stereo channel with same output
        sound = pygame.sndarray.make_sound(mu_out)
        channel = sound.play(-1)
        channel.set_volume(volLeft,volRight)
        pygame.time.delay(play_duration)
        sound.stop()

def sample_play(nnet,sample_rate):
    # when no midi device is plugged in this function will play random
    # latent variable instances to inspect the capabilities of the network
    buffer_size = 20
    play_duration = 900 # play duration in miliseconds
    pygame.mixer.pre_init(sample_rate, -16, 2,buffer = buffer_size) # 44.1kHz, 16-bit signed, stereo
    pygame.init()
    volLeft = 0.5;volRight=0.5 # Volume
    z_val = np.zeros([1,nnet.dim_z]) 
    while True:
        mu_out = nnet.generate(z_val)
        pdb.set_trace()
        mu_out = map_to_range(mu_out[4000:-4000],None,[-32768,32768],from_data=True)
        mu_out = mu_out.astype(np.int16)
        mu_out = np.array(zip(mu_out,mu_out)) # make stereo channel with same output
        sound = pygame.sndarray.make_sound(mu_out)
        channel = sound.play(-1)
        channel.set_volume(volLeft,volRight)
        pygame.time.delay(play_duration)
        sound.stop()
        z_val = np.random.uniform(-1,0,nnet.dim_z).astype(np.float32)

def trainset_play(nnet,sample_rate):
    # when no midi device is plugged in this function will play random
    # latent variable instances to inspect the capabilities of the network
    buffer_size = 20
    play_duration = 900 # play duration in miliseconds
    pygame.mixer.pre_init(sample_rate, -16, 2,buffer = buffer_size) # 44.1kHz, 16-bit signed, stereo
    pygame.init()
    volLeft = 0.5;volRight=0.5 # Volume
    z_val = np.zeros([1,nnet.dim_z]) # Initial latent variable values
    latents = nnet.get_latent_states(net.x_train)
    print latents
    while True:
        random_state = np.random.randint(0,latents.shape[0])
        mu_out = nnet.generate(latents[random_state,:])
        plt.plot(mu_out)
        plt.show()
        mu_out = map_to_range(mu_out,None,[-32768,32768],from_data=True)
        mu_out = mu_out.astype(np.int16)
        mu_out = np.array(zip(mu_out,mu_out)) # make stereo channel with same output
        sound = pygame.sndarray.make_sound(mu_out)
        channel = sound.play(-1)
        channel.set_volume(volLeft,volRight)
        pygame.time.delay(play_duration)
        sound.stop()
        mu_out = map_to_range(nnet.x_train[random_state,0,:,1],None,[-32768,32768],from_data=True)
        plt.plot(mu_out)
        pdb.set_trace()
        mu_out = mu_out.astype(np.int16)
        mu_out = np.array(zip(mu_out,mu_out)) # make stereo channel with same output

        sound = pygame.sndarray.make_sound(mu_out)
        channel = sound.play(-1)
        channel.set_volume(volLeft,volRight)
        pygame.time.delay(play_duration)
        sound.stop()
        plt.show()
        pygame.time.delay(1000)
        print "Done"

def sample_write(nnet,sample_rate,name ="default",duration=5,sample_for_latents = 20,retur=True):
    loops = sample_rate*duration/nnet.x_train.shape[2]
    print "LOOPS",loops
    idx = np.random.randint(0,nnet.x_train.shape[0],loops)
    idx2 = np.random.randint(0,nnet.x_train.shape[0],sample_for_latents)
    # get an idea of the latent space ranges:

    latents = nnet.get_latent_states(nnet.x_train[idx2])
    latent_min =np.amin(latents) ;latent_max =np.amax(latents)
    z_val = np.random.uniform(latent_min,latent_max,[loops,nnet.dim_z]).astype(np.float32)

    random_out = np.ravel(nnet.generate(z_val))
    random_out = np.int16(random_out/np.max(np.abs(random_out)) * 32767)
    trainset_out = np.ravel(nnet.output(nnet.x_train[idx]))
    trainset_out =np.int16(trainset_out/np.max(np.abs(trainset_out)) * 32767)
    original = np.ravel(nnet.x_train[idx])
    if retur==True:
        return random_out,trainset_out,original
    else:
        wav.write("sound/"+name+"_random.wav",sample_rate,random_out)
        wav.write("sound/"+name+"_trainset.wav",sample_rate,trainset_out)

def path_play(n_code, n_paths, n_steps=480):
    """
    create a random path through code space by interpolating between points
    """
    paths = []
    p_starts = np.random.randn(n_paths, n_code)
    for i in range(n_steps/48):
        p_ends = np.random.randn(n_paths, n_code)
        for weight in np.linspace(0., 1., 48):
            paths.append(p_starts*(1-weight) + p_ends*weight)
        p_starts = np.copy(p_ends)

    paths = np.asarray(paths)
    return paths

def path_write(nnet,sample_rate,duration=5,data_points = 2,name = "path_play",retur = True):
    loops = sample_rate*duration/nnet.x_train.shape[2]
    idx = np.random.randint(0,nnet.x_train.shape[0],data_points+1)
    latents = nnet.get_latent_states(nnet.x_train[idx])
    paths = []
    for i in range(data_points-1):
        for weight in np.linspace(0.,1.,loops):
            paths.append((latents[i,:]*(1-weight) + latents[i+1,:]*weight).astype(np.float32))
    paths = np.asarray(paths)
    random_out = np.ravel(nnet.generate(paths))
    random_out = np.int16(random_out/np.max(np.abs(random_out)) * 32767)
    original = np.ravel(nnet.x_train[idx])
    original = np.int16(original/np.max(np.abs(original)) * 32767)
    if retur == True:
        return random_out,original
    else:
        wav.write("sound/"+name+"_paths.wav",sample_rate,random_out)
        wav.write("sound/"+name+"_original.wav",sample_rate,original)
