from preprocess import pickle_loader,map_to_range_symmetric,map_to_range
import mido
import pygame
from VariationalAutoencoder import VA
import numpy as np
import theano
from convVAE2 import convVAE
import pdb


def  device_play(nnet,sample_rate,duration = 100,buffer_size=20):
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
    play_duration = 1000 # play duration in miliseconds
    pygame.mixer.pre_init(sample_rate, -16, 2,buffer = buffer_size) # 44.1kHz, 16-bit signed, stereo
    pygame.init()
    volLeft = 0.5;volRight=0.5 # Volume
    z_val = np.zeros(nnet.dim_z) # Initial latent variable values
    while True:
        print z_val
        mu_out = nnet.generate(z_val)
        mu_out = map_to_range(mu_out,None,[-32768,32768],from_data=True)
        mu_out = mu_out.astype(np.int16)
        mu_out = np.array(zip(mu_out,mu_out)) # make stereo channel with same output
        sound = pygame.sndarray.make_sound(mu_out)
        channel = sound.play(-1)
        channel.set_volume(volLeft,volRight)
        pygame.time.delay(play_duration)
        sound.stop()
        z_val = np.random.uniform(-20,20,nnet.dim_z)

def trainset_play(nnet,sample_rate):
    # when no midi device is plugged in this function will play random
    # latent variable instances to inspect the capabilities of the network
    buffer_size = 20
    play_duration = 1000 # play duration in miliseconds
    pygame.mixer.pre_init(sample_rate, -16, 2,buffer = buffer_size) # 44.1kHz, 16-bit signed, stereo
    pygame.init()
    volLeft = 0.5;volRight=0.5 # Volume
    z_val = np.zeros([1,nnet.dim_z]) # Initial latent variable values
    latents = nnet.get_latent_states(net.x_train)
    print latents
    pdb.set_trace()
    while True:
        random_state = np.random.randint(0,latents.shape[0])
        mu_out = nnet.generate(latents[random_state,:])
        mu_out = map_to_range(mu_out,None,[-32768,32768],from_data=True)
        mu_out = mu_out.astype(np.int16)
        mu_out = np.array(zip(mu_out,mu_out)) # make stereo channel with same output
        sound = pygame.sndarray.make_sound(mu_out)
        channel = sound.play(-1)
        channel.set_volume(volLeft,volRight)
        pygame.time.delay(play_duration)
        sound.stop()

        mu_out = map_to_range(nnet.x_train[random_state,0,0,:],None,[-32768,32768],from_data=True)
        mu_out = mu_out.astype(np.int16)
        mu_out = np.array(zip(mu_out,mu_out)) # make stereo channel with same output
        print mu_out
        sound = pygame.sndarray.make_sound(mu_out)
        channel = sound.play(-1)
        channel.set_volume(volLeft,volRight)
        pygame.time.delay(play_duration)
        sound.stop()
        pygame.time.delay(1000)
        print "Done"


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



if __name__ == "__main__":
    net = pickle_loader("times2.pkl")
    #lat = net.get_latent_states(net.x_train)
    #pdb.set_trace()
    sample_rate = 2000
    #device_play(net,sample_rate,duration=2000)
    sample_play(net,sample_rate)


