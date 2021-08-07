
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import scipy.io
from distances import *
from scipy.stats import wasserstein_distance


# Number of training pulses
NPULSESTRAIN = 5000
# Number of test pulses
NPULSESTEST = 5000
# Epochs of training
EPOCHS = 300
# Pulse length
LEN = 512
# Number of filters
FILT = 24
# Scale of the loaded pulses
SCALE = 1.0
# Pulse threshold
THR = 0.04 #0.09

# Batch size
BATCH_SIZE = 16
# Dimension of the pulses according to Generator
DIM = 256


# For histograms:

# Low-pass filter order (1 => no low-pass filter)
LPF = 1
# Pulse height max
SMAX = 0.5
# Tail amplitude of the pulse
SMIN = 0.15
# Bins of the histograms
BINS = 72


def CapturePulse(source, pos, pulselen, thr, leftmar=128, lpf=1):
    """ Extracts the pulse from the time sequence 'source' found from position 'pos'. The trigger threshold is 'thr'.
        The output pulse length is 'pulselen' and its left and right margin is 'leftmar'. """
    flag0, flag1 = 0, 0
    while (pos<len(source) and flag1==0):
        if flag0==0:
            if source[pos] < thr:
                flag0 = 1
        else:
            if source[pos] > thr:
                flag1 = 1
        pos += 1
    pulse = source[pos-leftmar:pos+pulselen-leftmar]
    pulseslice = np.copy(pulse)
    pulseslice[leftmar//2:pulselen//2] = 0
    

    if flag1==0 or any(i>thr for i in pulseslice):
        # Only accept pulses with minimum width
        return np.array([0])
    else:
        return np.convolve(pulse, np.ones(lpf)/lpf, 'same')

def CreateHistogram1D(real_pulses, synth_pulses, smax, thr, bins, lpf=1):
    if real_pulses.shape != synth_pulses.shape:
        raise ValueError('Shapes of arrays do not match')
    
    x_real_filt = np.zeros(real_pulses.shape)
    x_synth_filt = np.zeros(synth_pulses.shape)
    
    for i in range(0, real_pulses.shape[0]):
        x_synth_filt[i,:] = np.convolve(synth_pulses[i,:], np.ones(lpf)/lpf, 'same')
        x_real_filt[i,:] = np.convolve(real_pulses[i,:], np.ones(lpf)/lpf, 'same')
    
    x_real_max = np.max(x_real_filt,axis=1)
    x_synth_max = np.max(x_synth_filt,axis=1)
    
    # Real pulses histogram
    plt.figure(figsize=(4.5,3.5))
    histreal = plt.hist(x_real_max, bins, range=(thr, smax), histtype='bar', color='lightgreen', label='histreal')
    histsynth = plt.hist(x_synth_max, bins, range=(thr, smax), histtype='step', color='black', label='histsynth')
    plt.legend(['Real', 'Synthetized'], frameon=False)
    
    chi_dist = chisquared_distance(histreal[0],histsynth[0])
    w_dist = wasserstein_distance(histreal[0],histsynth[0])
    fwhm_dist = fwhm(histsynth)
    
    plt.xlabel('pulse height')
    plt.ylabel('counts')
    plt.grid(color='grey', linestyle=':', linewidth=1)
    plt.show()
    
    return chi_dist, fwhm_dist, w_dist
    

def CreateUniqueHistogram1D(real_pulses, smax, thr, bins, lpf=1):    
    x_real_filt = np.zeros(real_pulses.shape)
    
    N = real_pulses.shape[0]
    
    for i in range(0, N):
        x_real_filt[i,:] = np.convolve(real_pulses[i,:], np.ones(lpf)/lpf, 'same')
    
    x_real_max = np.max(x_real_filt,axis=1)
    
    # Real pulses histogram
    plt.figure(figsize=(4.5,3.5))
    hist = plt.hist(x_real_max, bins, range=(thr, smax), histtype='bar', color='blue', label='histreal')
    plt.xlabel('pulse height')
    plt.ylabel('counts')
    plt.show()

    return fwhm(hist)

def CreateHistogram2D(real_pulses, synth_pulses, smax, smin, thr, bins, lpf=1):
    if real_pulses.shape != synth_pulses.shape:
        raise ValueError('Shapes of arrays do not match')
    
    x_real_filt = np.zeros(real_pulses.shape)
    x_synth_filt = np.zeros(synth_pulses.shape)
    
    for i in range(0, real_pulses.shape[0]):
        x_synth_filt[i,:] = np.convolve(synth_pulses[i,:], np.ones(lpf)/lpf, 'same')
        x_real_filt[i,:] = np.convolve(real_pulses[i,:], np.ones(lpf)/lpf, 'same')
    
    x_real_max = np.max(x_real_filt,axis=1)
    x_synth_max = np.max(x_synth_filt,axis=1)
    x_real_min = np.min(x_real_filt,axis=1)
    x_synth_min = np.min(x_synth_filt,axis=1)
    
    # Real pulses histogram 2D
    plt.figure(figsize=(7,6.5))
    plt.subplot(2,1,1)
    plt.hist2d(x_real_max, x_real_min, bins, range=((0, smax),(-smin, 0)))
    plt.grid(color='black', linestyle='solid', linewidth=0.5)
    plt.title('Real pulses')
    plt.ylabel('min (tail)')
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 10
    cbar.ax.set_ylabel('number of pulses', rotation=270)
    
    # Generated pulses histogram 2D
    plt.subplot(2,1,2)
    plt.hist2d(x_synth_max, x_synth_min, bins, range=((0, smax),(-smin, 0)))
    plt.grid(color='black', linestyle='solid', linewidth=0.5)
    plt.title('Synth pulses')
    plt.xlabel('max (peak)')
    plt.ylabel('min (tail)')
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 10
    cbar.ax.set_ylabel('number of pulses', rotation=270)
    plt.show()

def PlotExamplePulses(real_pulses, synth_pulses):    
    plt.figure(figsize=(7,6))
    plt.subplot(2,1,1)
    plt.plot(real_pulses)
    plt.ylabel(r'Amplitude [V]')
    plt.title('Real pulses')
    plt.grid(color='grey', linestyle=':', linewidth=1)
    plt.subplot(2,1,2)
    plt.plot(synth_pulses)
    plt.ylabel(r'amplitude [V]')
    plt.xlabel(r'time [$\mu$s]')
    plt.title('Synth pulses')
    plt.grid(color='grey', linestyle=':', linewidth=1)
    plt.show()

def ExtractMinMax(pulses):
    n = pulses.shape[0]
    minmax = np.zeros((n, 2))
    for i in range(0, n):
        minmax[i, 0] = np.min(pulses[i, :])
        minmax[i, 1] = np.max(pulses[i, :])
    return minmax

def distance_between_pulses(pulse1, pulse2, order=2):
    return np.sum(np.abs((pulse1 - pulse2)**order))

def distance_between_max_pulses(pulse1, pulse2, order=2):
    return np.abs(np.max(pulse1) - np.max(pulse2))**order

def distance_between_min_pulses(pulse1, pulse2, order=2):
    return np.abs(np.min(pulse1) - np.min(pulse2))**order
        
def select_pulses(pulses, interval):
    n = pulses.shape[0]
    max_val = np.zeros(n)
    for i in range(0, n):
        max_val[i] = np.max(pulses[i, :])
    
    loi = []
    for j in range(0, n):
        if max_val[j] >= interval[0] and max_val[j] <= interval[1]:
            loi.append(j)
    return pulses[loi]
    
def CompDistances(pulses, slots=20, max_height=1, dist_func=distance_between_pulses, order=2):
    interv = np.linspace(0, max_height, slots+1)
    
    num_pulses_slot = np.zeros(slots)
    dist_t = []
    for i in range(1, slots+1):
        sel_pulses = select_pulses(pulses, (interv[i-1], interv[i]))
        
        # Calculate distances among them
        num_sel_pulses = len(sel_pulses)
        dist1a = []
        for a1 in range(0, num_sel_pulses):
            for a2 in range(a1 + 1, num_sel_pulses):
                dist1a.append(dist_func(sel_pulses[a1], sel_pulses[a2], order))
    
        dist_t.append(dist1a)
        num_pulses_slot[i-1] = num_sel_pulses
    return dist_t, num_pulses_slot

def CompDistancesBetweenGroups(pulses_g1, pulses_g2, slots=20, max_height=1, dist_func=distance_between_pulses, order=2):
    interv = np.linspace(0, max_height, slots+1)
    
    num_pulses_g1_slot = np.zeros(slots)
    num_pulses_g2_slot = np.zeros(slots)
    dist_t = []
    for i in range(1, slots+1):
        sel_pulses_g1 = select_pulses(pulses_g1, (interv[i-1], interv[i]))
        sel_pulses_g2 = select_pulses(pulses_g2, (interv[i-1], interv[i]))

        # Calculate distances among them
        num_sel_pulses_g1 = len(sel_pulses_g1)
        num_sel_pulses_g2 = len(sel_pulses_g2)
        dist1a = []
        for a1 in range(0, num_sel_pulses_g1):
            for a2 in range(0, num_sel_pulses_g2):
                dist1a.append(dist_func(sel_pulses_g1[a1], sel_pulses_g2[a2], order))
        dist_t.append(dist1a)
        num_pulses_g1_slot[i-1] = num_sel_pulses_g1
        num_pulses_g2_slot[i-1] = num_sel_pulses_g2
        
    return dist_t, num_pulses_g1_slot, num_pulses_g2_slot


def CompAndPlotDistances(real_pulses, synth_pulses, slots, max_height=1, dist_func=distance_between_pulses, order=2):
    print("Calculating distances beetween real pulses...")
    ld_real, _ = CompDistances(real_pulses, slots, max_height, dist_func, order)
    print("Calculating distances beetween synth pulses...")
    ld_synth, _ = CompDistances(synth_pulses, slots, max_height, dist_func, order)
    print("Calculating distances beetween real and synth pulses...")
    ld_real_synth, _, _ = CompDistancesBetweenGroups(real_pulses, synth_pulses, slots, max_height, dist_func, order)
    
    dist = np.zeros((3, slots, 2))
    
    for n in range(0, slots):
        if ld_real[n]:
            dist[0, n, :] = np.mean(ld_real[n]), np.std(ld_real[n])
        if ld_synth[n]:
            dist[1, n, :] = np.mean(ld_synth[n]), np.std(ld_synth[n])
        if ld_real_synth[n]:
            dist[2, n, :] = np.mean(ld_real_synth[n]), np.std(ld_real_synth[n])
    
    x_real = np.linspace(0, max_height, slots)
    x_synth = np.linspace(0, max_height, slots)
    x_real_synth = np.linspace(0, max_height, slots)
    
    plt.figure()
    #plt.errorbar(x_real, dist[0, :, 0], yerr=dist[0, :, 1], color='orange', fmt='o')
    #plt.errorbar(x_synth, dist[1, :, 0], yerr=dist[1, :, 1], color='indigo', fmt='o')
    #plt.errorbar(x_real_synth, dist[2, :, 0], yerr=dist[2, :, 1], color='green', fmt='o')
    plt.plot(x_real, dist[0, :, 0], color='orange')
    plt.plot(x_synth, dist[1, :, 0], color='indigo')
    plt.plot(x_real_synth, dist[2, :, 0], color='green')

    plt.legend(('Among real pulses', 'Among synth. pulses', 'Among real and synth.'), frameon=False)
    plt.xlabel('pulse height')
    plt.ylabel('distance')
    plt.grid(color='grey', linestyle=':', linewidth=1)
    plt.show()
        
def CompAndPlotMaxMin(real_pulses, synth_pulses):
    minmax_real = ExtractMinMax(real_pulses)
    minmax_synth = ExtractMinMax(synth_pulses)
    
    plt.scatter(minmax_synth[:,0], minmax_synth[:,1], color='blue', marker='.', alpha=0.05)
    plt.scatter(minmax_real[:,0], minmax_real[:,1], color='orange', marker='.', alpha=0.05)
    plt.grid(color='grey', linestyle=':', linewidth=1)
    plt.show()


# Get train samples
mat = scipy.io.loadmat('./source1.mat')

source = np.array(mat['Cs137_4usdiv_1Vdiv_002'])  # Recommended threshold = 0.04
#source = np.array(mat['Na22_4usdiv_1Vdiv_002'])  # Recommended threshold = 0.09

source = (source.T)[:,0]
source = source / np.max(source)
   
LPF = 4

x_train = np.zeros((NPULSESTRAIN, LEN, 1), dtype=np.float32)
i = 0
while i < NPULSESTRAIN:
    sys.stdout.write('\rPrepared {}/{} samples for train'.format(i + 1, NPULSESTRAIN))

    pos = np.random.randint(LEN//3,len(source)-(LEN*2))
    pulse = CapturePulse(source, pos, LEN, THR, lpf=LPF) * SCALE
    if len(pulse)==LEN:
        x_train[i,:,0] = pulse
        i += 1

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(NPULSESTRAIN).batch(BATCH_SIZE)

# Get test samples
x_test = np.zeros((NPULSESTEST, LEN, 1), dtype=np.float32)
i = 0
while i < NPULSESTEST:
    sys.stdout.write('\rPrepared {}/{} samples for test.'.format(i + 1, NPULSESTEST))

    pos = np.random.randint(LEN//3,len(source)-(LEN*2))
    pulse = CapturePulse(source, pos, LEN, THR, lpf=LPF) * SCALE
    if len(pulse)==LEN:
        x_test[i,:,0] = pulse
        i += 1

%matplotlib inline

def make_generator_model():
    """ Creates the Generator """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(DIM,)),
            tf.keras.layers.Dense(units=64*48, activation=tf.nn.leaky_relu),
            tf.keras.layers.Reshape(target_shape=(64, 48)),
            tf.keras.layers.Conv1D(filters=48, kernel_size=6, padding='same', activation=tf.nn.leaky_relu),
            tf.keras.layers.UpSampling1D(2),
            tf.keras.layers.Conv1D(filters=48, kernel_size=6, padding='same', activation=tf.nn.leaky_relu),
            tf.keras.layers.UpSampling1D(2),
            tf.keras.layers.Conv1D(filters=48, kernel_size=6, padding='same', activation=tf.nn.leaky_relu),
            tf.keras.layers.UpSampling1D(2),
            tf.keras.layers.Conv1D(filters=48, kernel_size=6, padding='same', activation=tf.nn.leaky_relu),
            tf.keras.layers.Conv1D(filters=1, kernel_size=7, padding='same', activation='linear'),
        ]
    )
    return model

def make_discriminator_model():
    """ Creates the Discriminator """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(LEN,1)),
            tf.keras.layers.Conv1D(filters=16, kernel_size=6, strides=2, padding='same', activation=tf.nn.leaky_relu),
            tf.keras.layers.Conv1D(filters=32, kernel_size=6, strides=2, padding='same', activation=tf.nn.leaky_relu),
            tf.keras.layers.Conv1D(filters=64, kernel_size=6, strides=2, padding='same', activation=tf.nn.leaky_relu),
            tf.keras.layers.Conv1D(filters=128, kernel_size=6, strides=2, padding='same', activation=tf.nn.leaky_relu),
            tf.keras.layers.Conv1D(filters=256, kernel_size=6, strides=2, padding='same', activation=tf.nn.leaky_relu),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ]
    )

    return model

generator = make_generator_model()
generator.summary()

discriminator = make_discriminator_model()
discriminator.summary()


loss_func = tf.keras.losses.BinaryCrossentropy()

def discriminator_loss(real_output, synth_output):
    real_loss = loss_func(tf.ones_like(real_output), real_output)
    synth_loss = loss_func(tf.zeros_like(synth_output), synth_output)
    total_loss = real_loss + synth_loss
    return total_loss/2

def discriminator_loss_real(real_output):
    return loss_func(tf.ones_like(real_output), real_output)


def discriminator_loss_synth(synth_output):
    return loss_func(tf.zeros_like(synth_output), synth_output)

def generator_loss(synth_output):
    return loss_func(tf.ones_like(synth_output), synth_output)

generator_optimizer = tf.keras.optimizers.Adam(lr=0.00004, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(lr=0.00004, beta_1=0.5)

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, DIM])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)
      real_output = discriminator(images, training=True)
      synth_output = discriminator(generated_images, training=True)
      gen_loss = generator_loss(synth_output)
            
      disc_loss_real = discriminator_loss_real(real_output)
      disc_loss_synth = discriminator_loss_synth(synth_output)
      
      disc_loss = (disc_loss_real + disc_loss_synth) / 2
        
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss_real, disc_loss_synth
    
def train(dataset, epochs, x_test):
  g_loss = list()
  dr_loss = list()
  df_loss = list()
  chi_dist = np.zeros(epochs)
  pulse_dist = np.zeros((epochs, 2))
  f_dist = np.zeros(epochs)
  w_dist = np.zeros(epochs)
    
  for epoch in range(epochs):
    start = time.time()
    dataset = dataset.shuffle(NPULSESTRAIN)

    for image_batch in dataset:
        gen_loss, disc_loss_real, disc_loss_synth = train_step(image_batch)

        g_loss_tmp = gen_loss.numpy()
        dr_loss_tmp = disc_loss_real.numpy()
        df_loss_tmp = disc_loss_synth.numpy()
        
        d_loss_tmp = (dr_loss_tmp + df_loss_tmp)/2
        
        g_loss.append(g_loss_tmp)
        dr_loss.append(dr_loss_tmp)
        df_loss.append(df_loss_tmp)
        
        sys.stdout.write('\rgen_loss={:.4f} discr_loss={:.4f} (real={:.4f} synth={:.4f}) '.format(g_loss_tmp, d_loss_tmp, dr_loss_tmp, df_loss_tmp))

    sys.stdout.write('\nTime for epoch {} is {:.2f} sec\n'.format(epoch + 1, time.time()-start))

    real_pulses = x_test[:,:,0]
    synth_pulses = generator(tf.random.normal([NPULSESTEST,DIM])).numpy()
    synth_pulses = synth_pulses[...,0]

    # Calculate distance between real andh synth pulses
    N = 500
    d_rs, _, _ = CompDistancesBetweenGroups(real_pulses[0:N], synth_pulses[0:N], BINS//2, max_height=SMAX, dist_func=distance_between_min_pulses, order=2)
    
    flat_list = [item for sublist in d_rs for item in sublist]
    pulse_dist[epoch, 0] = np.mean(np.array(flat_list))
    pulse_dist[epoch, 1] = np.std(np.array(flat_list))
    
    # Calculate distance between histograms
    chd, fw, wd = CreateHistogram1D(real_pulses, synth_pulses, SMAX, THR, BINS)
    
    chi_dist[epoch] = chd
    f_dist[epoch] = fw
    w_dist[epoch] = wd

    # Plot example
    N = 10
    PlotExamplePulses(real_pulses[0:N].T, synth_pulses[0:N].T)

  return g_loss, dr_loss, df_loss, chi_dist, np.array(pulse_dist), f_dist, w_dist
  
g_loss, dr_loss, df_loss, chi_dist, pulse_dist, f_dist, w_dist = train(train_dataset, EPOCHS, x_test)

# Load pretrained network and variables from Colab if applicable


#generator.save_weights('my_w_generator.h5')
#discriminator.save_weights('my_w_discriminator.h5')

generator.load_weights('my_w_generator.na.h5')
discriminator.load_weights('my_w_discriminator.na.h5')

##############################################################################
# Plot results

# Latex style
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size': 11,
    'text.usetex': True,
    'pgf.rcfonts': False,})

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

%matplotlib qt5

# Plot pulses

# Number of pulses to plot
N = 10

real_pulses = x_train[0:N,:,0].T
synth_pulses = generator(tf.random.normal([N,DIM])).numpy()
synth_pulses = synth_pulses[...,0].T

PlotExamplePulses(real_pulses, synth_pulses)


# Number of pulses to calculate the pulse mean
NM = 20

real_pulses1 = x_train[0:NM,:,0].T
synth_pulses1 = generator(tf.random.normal([N,DIM])).numpy()
synth_pulses1 = synth_pulses1[...,0].T

real_pulse_e1 = np.mean(real_pulses1,axis=1)
synth_pulse_e1 = np.mean(synth_pulses1,axis=1)

plt.figure(figsize=(4,7))
plt.subplot(2,1,1)  
plt.specgram(real_pulse_e1,Fs=1,cmap=plt.cm.get_cmap('jet'))
plt.ylabel('Frequency [Hz]')
plt.title('Real pulses')
cbar = plt.colorbar()
cbar.ax.get_yaxis().labelpad = 10
cbar.ax.set_ylabel('amplitude', rotation=270)

plt.subplot(2,1,2)
plt.specgram(synth_pulse_e1,Fs=1,cmap=plt.cm.get_cmap('jet'))
plt.xlabel('time [ns]')
plt.ylabel('Frequency [Hz]')
plt.title('Synthetized pulses')
cbar = plt.colorbar()
cbar.ax.get_yaxis().labelpad = 10
cbar.ax.set_ylabel('amplitude', rotation=270)
plt.show()


# Learn curve

LPFLR = 200
xrange = np.arange(0, len(g_loss)-LPFLR)

g_loss_i = np.convolve(g_loss, (1/LPFLR)*np.ones(LPFLR), 'same')
dr_loss_i = np.convolve(dr_loss, (1/LPFLR)*np.ones(LPFLR), 'same')
df_loss_i = np.convolve(df_loss, (1/LPFLR)*np.ones(LPFLR), 'same')

plt.figure(figsize=(6,4))

line_g, = plt.plot(xrange, g_loss_i[:-LPFLR], color='blue', linewidth=1)
line_dr, = plt.plot(xrange, dr_loss_i[:-LPFLR], color='green', linewidth=1)
line_df, = plt.plot(xrange, df_loss_i[:-LPFLR], color='red', linewidth=1)
plt.grid(color='grey', linestyle=':', linewidth=1)
plt.xlim(0,len(g_loss)-LPFLR)
#plt.ylim(0.65,0.8)
plt.legend((line_g, line_dr, line_df), ('generator loss', 'discriminator loss (real)', 'discriminator loss (synth)'), frameon=False)
plt.xlabel('Epoch')

ticks = np.arange(0,EPOCHS+1,step=50)
ticks2 = ticks * (len(g_loss)//EPOCHS)
plt.xticks(ticks2, ticks)
plt.show()



# Histograms

real_pulses = x_test[:,:,0]
synth_pulses = generator(tf.random.normal([NPULSESTEST,DIM])).numpy()
synth_pulses = synth_pulses[...,0]

CreateHistogram1D(real_pulses, synth_pulses, SMAX, THR, BINS)
CreateHistogram2D(real_pulses, synth_pulses, SMAX, SMIN, THR, BINS//2)

# MinMaxCorr

CompAndPlotMaxMin(real_pulses, synth_pulses)

# Distances

epo = np.arange(1, EPOCHS+1)

plt.figure()
plt.subplot(3,1,1)
plt.tight_layout(0.10)
plt.semilogy(epo, pulse_dist[:,0])
plt.fill_between(epo, pulse_dist[:,0]-pulse_dist[:,1], pulse_dist[:,0]+pulse_dist[:,1], alpha=0.4, linewidth=1)
plt.grid(color='grey', linestyle=':', linewidth=1)
plt.ylabel('Distance')
plt.title("Distance between real and synth pulses")

plt.subplot(3,1,2)
plt.tight_layout(0.10)
line_chi, = plt.semilogy(epo, chi_dist/100, color='blue', linewidth=1)
line_w, = plt.semilogy(epo, w_dist, color='green')
plt.ylabel('Distance')
plt.legend((line_chi, line_w), (r'${\chi}^{2}/100$', 'Wasserstein'), frameon=False)
plt.grid(color='grey', linestyle=':', linewidth=1)
plt.title("Distance between real and synth histograms")


plt.figure()
histreal = plt.hist(np.max(real_pulses, axis=1), BINS, range=(THR, SMAX))
plt.close()

plt.subplot(3,1,3)
plt.tight_layout(0.10)
fwhm_real_dist = fwhm(histreal)
f_real = (epo>-1)*fwhm_real_dist
plt.plot(epo[120:], f_dist[120:], color='black', linewidth=1)
plt.plot(epo, f_real, 'k--', linewidth=1)
plt.legend(('Generated pulses', 'Real pulses'), frameon=False)
plt.grid(color='grey', linestyle=':', linewidth=1)
plt.xlabel('epoch')
plt.ylabel('FWHM')
plt.title("FWHM of histograms")
plt.show()


# Distances

CompAndPlotDistances(real_pulses, synth_pulses, slots = BINS//2, max_height=SMAX, dist_func=distance_between_pulses, order=2)
CompAndPlotDistances(real_pulses, synth_pulses, slots = BINS//2, max_height=SMAX, dist_func=distance_between_max_pulses, order=2)
CompAndPlotDistances(real_pulses, synth_pulses, slots = BINS//2, max_height=SMAX, dist_func=distance_between_min_pulses, order=2)

