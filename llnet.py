import os
import sys
import time
import cPickle
import numpy
import h5py
import scipy.io
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import PIL.Image
import shutil
import Data_process2

from easygui import *

from theano.tensor.shared_randomstreams import RandomStreams
from logistic_sgd import LogisticRegression, load_data, load_data_overlapped, load_data_overlapped_strides
from mlp import HiddenLayer
from dA import dA
from utils import tile_raster_images
from Data_process2 import reconstruct_from_patches_with_strides_2d
from sklearn.feature_extraction import image as im
from scipy import misc, ndimage
from skimage import color, data, restoration
import nlinalg

#####################################################################################################################
#                                                                                                                   #
#                                                 Training Code                                                     #
#                                                                                                                   #
#####################################################################################################################

#######################################
#     Hyperparameters / Options       #
#######################################

# Training Dataset
# tr_dataset = 'dataset/20151026_lowlightnoisy_17x17.mat'

# Hyperparameters
patch_size = (17,17)
prod = patch_size[0]*patch_size[1]
hp_hlsize = [1000,1000,1000,1000,1000]
hp_corruption_levels = [0.1, 0.1, 0.1, 0.1, 0.1]
hp_pretraining_epochs = 3
hp_batchsize = 10 #llnet1: 50

#######################################
#          Class Construction         #
#######################################

class SdA(object):

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        n_ins=prod,
        hidden_layers_sizes=[500,500],
        n_outs=prod,
        corruption_levels=[0.1, 0.1]
    ):

        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
            
        self.x = T.matrix('x')
        self.y = T.matrix('y')
        
        for i in xrange(self.n_layers):
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)

            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)

        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs
        )

        self.params.extend(self.logLayer.params)
        self.finetune_cost = self.logLayer.image_norm(self.y, obj=self)
        self.errors = self.logLayer.image_norm(self.y, obj=self)

    def pretraining_functions(self, train_set_x, batch_size):

        index = T.lscalar('index')
        corruption_level = T.scalar('corruption')
        learning_rate = T.scalar('lr')
        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)

            fn = theano.function(
                inputs=[
                    index,
                    theano.Param(corruption_level, default=0.2),
                    theano.Param(learning_rate, default=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, train_set,valid_set,test_set,batch_size, learning_rate):

        (train_set_x, train_set_y) = train_set
        (valid_set_x, valid_set_y) = valid_set
        (test_set_x, test_set_y) = test_set

        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')

        gparams = T.grad(self.finetune_cost, self.params)

        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test'
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )

        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score

#######################################
#             SDA Training            #
#######################################

def test_SdA(finetune_lr=0.1, pretraining_epochs=hp_pretraining_epochs,
             pretrain_lr=0.1, training_epochs=100000, batch_size=hp_batchsize, patch_size = patch_size):
    
    datasets = load_data(tr_dataset)
    train_set = datasets[0]
    valid_set = datasets[1]
    test_set = datasets[2]
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    datasets = []

    print '... plotting clean images'
    image = PIL.Image.fromarray(tile_raster_images(
	X=test_set_y.get_value(),
	img_shape=patch_size, tile_shape=(50, 40),
	tile_spacing=(0, 0),scale_rows_to_unit_interval=False))
    image.save('outputs/LLnet_clean.png')

    print '... plotting noisy images'
    image = PIL.Image.fromarray(tile_raster_images(
	X=test_set_x.get_value(),
	img_shape=patch_size, tile_shape=(50, 40),
	tile_spacing=(0, 0),scale_rows_to_unit_interval=False))
    image.save('outputs/LLnet_noisy.png')
    
    n_train_samples = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches = n_train_samples/batch_size

    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'

    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=patch_size[0]*patch_size[1],
        hidden_layers_sizes= hp_hlsize,
        n_outs=patch_size[0]*patch_size[1]
    )

    print '... compiling functions'

    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_y,
                                                batch_size=batch_size)

    print '... pre-training the model'
    start_time = time.clock()
    for i in xrange(sda.n_layers):
	if i <= sda.n_layers/2:
	
	    if i == (sda.n_layers - 1):
	        currentlr = pretrain_lr;
	    else:
	        currentlr = pretrain_lr*0.1
	        
	    for epoch in xrange(pretraining_epochs):
	        c = []
	        for batch_index in xrange(n_train_batches):
	            current_c = pretraining_fns[i](index=batch_index,
	                     corruption=hp_corruption_levels[i],
	                     lr=currentlr)
	            if (batch_index % (n_train_batches/100 + 1) == 0):
	                print '    ... Layer %i Epoch %i Progress %i/%i, Cost: %.4f, AvgCost: %.4f' %(i, epoch, batch_index, n_train_batches, current_c, numpy.mean(c))
	            c.append(current_c)
	        print 'Pre-trained layer %i, epoch %d, cost ' % (i, epoch),
	        print numpy.mean(c)
	           
	        print '     model checkpoint for current epoch...'
		f = file('outputs/model_checkpoint.obj', 'wb')
		cPickle.dump(sda,f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()

    end_time = time.clock()
	                  
    print ('... pretrained bottom half of the SdA in %.2fm' % ((end_time - start_time) / 60.))
    
    
    layer_all = sda.n_layers + 1 #Number of hidden layers + 1
    print layer_all
    
    for i in xrange(layer_all/2 - 1):
	    
	    #Reverse map 2 to 5
	    layer = i+2
	    layer_applied = layer_all - layer + 1
	    print '... applying weights from SdA layer', layer, 'to SdA layer', (layer_applied)
	    ww, bb, bbp = [sda.dA_layers[layer-1].W.get_value(), sda.dA_layers[layer-1].b.get_value(), sda.dA_layers[layer-1].b_prime.get_value()]
	    sda.dA_layers[layer_applied-1].W.set_value(ww.T)
	    sda.dA_layers[layer_applied-1].b.set_value(bbp)
	    sda.dA_layers[layer_applied-1].b_prime.set_value(bb)
	    
    #Reverse map 1 to loglayer
    layer = 1
    print '... applying weights from SdA layer', layer, 'to loglayer layer'
    ww, bb, bbp = [sda.dA_layers[layer-1].W.get_value(), sda.dA_layers[layer-1].b.get_value(), sda.dA_layers[layer-1].b_prime.get_value()]
    sda.logLayer.W.set_value(ww.T)
    sda.logLayer.b.set_value(bbp)
    
    
    '''#Set sigmoid layer weights equal to dA weights
    for i in xrange(sda.n_layers):
	sda.sigmoid_layers[i].W.set_value(sda.dA_layers[i].W.get_value())
	sda.sigmoid_layers[i].b.set_value(sda.dA_layers[i].b.get_value())'''
    

    print '... compiling functions'
    train_fn, validate_model, test_model = sda.build_finetune_functions(
        train_set = train_set,valid_set = valid_set,test_set=test_set,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )
    
    reconstructed  = theano.function([],
    sda.logLayer.y_pred,givens={
	sda.x: test_set_x},on_unused_input='ignore')
	
    w1  = theano.function([],
    nlinalg.trace(T.dot(sda.sigmoid_layers[0].W.T,sda.sigmoid_layers[0].W)),givens={
	sda.x: test_set_x},on_unused_input='ignore')
	
    w2  = theano.function([],
    nlinalg.trace(T.dot(sda.sigmoid_layers[1].W.T,sda.sigmoid_layers[1].W)),givens={
	sda.x: test_set_x},on_unused_input='ignore')
	
    w3  = theano.function([],
    nlinalg.trace(T.dot(sda.sigmoid_layers[2].W.T,sda.sigmoid_layers[2].W)),givens={
	sda.x: test_set_x},on_unused_input='ignore')
	
    w4  = theano.function([],
    nlinalg.trace(T.dot(sda.sigmoid_layers[3].W.T,sda.sigmoid_layers[3].W)),givens={
	sda.x: test_set_x},on_unused_input='ignore')
	
    w5  = theano.function([],
    nlinalg.trace(T.dot(sda.sigmoid_layers[4].W.T,sda.sigmoid_layers[4].W)),givens={
	sda.x: test_set_x},on_unused_input='ignore')
	
    wl  = theano.function([],
    nlinalg.trace(T.dot(sda.logLayer.W.T,sda.logLayer.W)),givens={
	sda.x: test_set_x},on_unused_input='ignore')
		
    '''print '     loading previous model...'
    f = file('outputs/model_bestpsnr.obj', 'rb')
    sda = cPickle.load(f)
    f.close()'''
    
    print '... finetuning the model'
    patience = 100000 * n_train_batches
    patience_increase = 2.
    improvement_threshold = 1
    validation_frequency = min(n_train_batches, patience / 2)
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()
    done_looping = False
    epoch = 0

    plot_valid_error = []
    ww1 = []
    ww2 = []
    ww3 = []
    ww4 = []
    ww5 = []
    wwl = []
    psnrs = []
    best_psnr = []
    
    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        
        if 1 == 0: ########################################################################## Switch for on-the-fly training data generation
        
		if epoch % 50 == 0:
		    print '... calling matlab function!'
		    call(["/usr/local/MATLAB/R2015a/bin/matlab","-nodesktop","-r",'end2end_datagen_256; exit'])
		    print '... data regeneration complete, loading new data'
		    datasets = load_data('dataset/llnet_17x17_OTF.mat')
		    train_set = datasets[0]
		    valid_set = datasets[1]
		    test_set = datasets[2]
		    train_set_x, train_set_y = datasets[0]
		    valid_set_x, valid_set_y = datasets[1]
		    test_set_x, test_set_y = datasets[2]
		    datasets = []
		    
		    reconstructed  = theano.function([],
		    sda.logLayer.y_pred,givens={
			sda.x: test_set_x},on_unused_input='warn')
		    
		    print '... plotting clean images'
		    image = PIL.Image.fromarray(tile_raster_images(
			X=test_set_y.get_value(),
			img_shape=patch_size, tile_shape=(50, 40),
			tile_spacing=(0, 0),scale_rows_to_unit_interval=False))
		    image.save('outputs/LLnet_clean.png')

		    print '... plotting noisy images'
		    image = PIL.Image.fromarray(tile_raster_images(
			X=test_set_x.get_value(),
			img_shape=patch_size, tile_shape=(50, 40),
			tile_spacing=(0, 0),scale_rows_to_unit_interval=False))
		    image.save('outputs/LLnet_noisy.png')

        if 1 == 1: ########################################################################## Switch for training rate schedule change
        
		if epoch % 200 == 0:
		    tempval = finetune_lr * 0.1
		    print '... switching learning rate to %.4f, recompiling function'%(tempval)
		    train_fn, validate_model, test_model = sda.build_finetune_functions(
			train_set = train_set,valid_set = valid_set,test_set=test_set,
			batch_size=batch_size,
			learning_rate=tempval
		    )
        
        for minibatch_index in xrange(n_train_batches): 
            minibatch_avg_cost = train_fn(minibatch_index)
            if (minibatch_index % (n_train_batches/100 + 1) == 0):
	        print '    ... FT E%i, %i/%i/%i, aCost: %.4f' %(epoch, minibatch_index, n_train_batches, hp_batchsize, minibatch_avg_cost)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation loss %f (best: %f)' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss, best_validation_loss))
                       
                plot_valid_error.append(this_validation_loss)

		# Training monitoring tools -----------------------------------------

		ww1.append(w1())
		ww2.append(w2())
		ww3.append(w3())
		ww4.append(w4())
		ww5.append(w5())
		wwl.append(wl())
		
		psnr = 10*numpy.log10(255**2 / numpy.mean(numpy.sqrt(numpy.sum(((test_set_y.get_value() - reconstructed())*255)**2,axis=1,keepdims=True))))
		psnrs.append( psnr )
		
		if psnr >= numpy.max(psnrs):
	            print '     saving trained model based on highest psnr...'
		    f = file('outputs/model_bestpsnr.obj', 'wb')
		    cPickle.dump(sda,f, protocol=cPickle.HIGHEST_PROTOCOL)
		    f.close()
                    print '     plotting reconstructed images based on highest psnr...'
                    image = PIL.Image.fromarray(tile_raster_images(
		    X=reconstructed(),
		    	img_shape=patch_size, tile_shape=(50,40),
		    	tile_spacing=(0, 0),scale_rows_to_unit_interval=False))
                    image.save('outputs/LLnet_reconstructed_bestpsnr.png')
		
                plt.clf()
                plt.suptitle('Epoch %d'%(epoch))
                plt.subplot(121); plt.plot(plot_valid_error,'-xb'); plt.title('Validation Error, best %.4f'%(numpy.min(plot_valid_error)))  
                plt.subplot(122); plt.plot(psnrs,'-xb'); plt.title('PSNR, best %.4f dB'%(numpy.max(psnrs)));
                if len(psnrs)>2:
                    plt.xlabel('Rate: %.4f dB/step'%(psnrs[-1] - psnrs[-2]))
                plt.savefig('outputs/validation_error.png')
		
		
		plt.clf()
		plt.suptitle('Weight Norms, epoch %d'%(epoch))
		plt.subplot(231); plt.plot(ww1,'-xr'); plt.axis('tight'); plt.title('Layer1')
		plt.subplot(232); plt.plot(ww2,'-xc'); plt.axis('tight'); plt.title('Layer2')
		plt.subplot(233); plt.plot(ww3,'-xy'); plt.axis('tight'); plt.title('Layer3')
		plt.subplot(234); plt.plot(ww4,'-xg'); plt.axis('tight'); plt.title('Layer4')
		plt.subplot(235); plt.plot(ww5,'-xb'); plt.axis('tight'); plt.title('Layer5')
		plt.subplot(236); plt.plot(wwl,'-xm'); plt.axis('tight'); plt.title('Sigmoid Layer')
		plt.savefig('outputs/weightnorms.png')
		
		# Training monitoring tools -----------------------------------------

                if this_validation_loss < best_validation_loss:
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test loss of '
                           'best model %f') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score))
			    
	            print '     saving trained model based on lowest validation error...'
		    f = file('outputs/model.obj', 'wb')
		    cPickle.dump(sda,f, protocol=cPickle.HIGHEST_PROTOCOL)
		    f.close()
               
                    print '     plotting reconstructed images...'
                    image = PIL.Image.fromarray(tile_raster_images(
		    X=reconstructed(),
		    	img_shape=patch_size, tile_shape=(50,40),
		    	tile_spacing=(0, 0),scale_rows_to_unit_interval=False))
                    image.save('outputs/LLnet_reconstructed.png')
		    
		    print '     plotting complete. Training next epoch...'

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(
        (
            'Optimization complete with best validation loss of %f, '
            'on iteration %i, '
            'with test performance %f'
        )
        % (best_validation_loss, best_iter + 1, test_score)
    )
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

#####################################################################################################################
#                                                                                                                   #
#                                                Inference Code                                                     #
#                                                                                                                   #
#####################################################################################################################
    
######################################################
#    Overlapping Patches Denoising (With Strides)    #  (Default)
######################################################

def denoise_overlapped_strides(strides=(3,3)): #1 2 4 11

    #print '=== OVERLAPPING PATCHES',strides,'STRIDES ==============================='
    
    testdata = misc.imread(te_noisy_image,flatten=True)
    fname=te_noisy_image.rsplit('/',1)[-1][:-4]
    #scipy.misc.imsave('outputs/LLnet_inference_'+fname+'_test.png',testdata)
    shutil.copyfile(te_noisy_image, 'outputs/ori_'+fname+'.png')
    
    test_set_x, te_h,te_w = load_data_overlapped_strides(te_dataset = te_noisy_image, patch_size = patch_size, strides=strides)
    im_ = test_set_x.get_value()
    im_noisy = im_.reshape((im_).shape[0], *patch_size)
    rec_n = im.reconstruct_from_patches_2d(im_noisy, (te_h,te_w)) 

    reconstructed  = theano.function([],
        sda.logLayer.y_pred,givens={
	sda.x: test_set_x},on_unused_input='warn')
    result = reconstructed()

    im_recon = result.reshape((result).shape[0], *patch_size)
    rec_r = reconstruct_from_patches_with_strides_2d(im_recon, (te_h,te_w), strides=strides) 

    scipy.misc.imsave('outputs/LLnet_inference_'+fname+'_out.png',rec_r)
    
#    print sda.sigmoid_layers[0].W.get_value().shape
#    print sda.sigmoid_layers[1].W.get_value().shape
#    print sda.sigmoid_layers[2].W.get_value().shape
#    print sda.sigmoid_layers[3].W.get_value().shape
#    print sda.sigmoid_layers[4].W.get_value().shape
#    print sda.sigmoid_layers[5].W.get_value().shape
#    print sda.sigmoid_layers[6].W.get_value().shape
    
    filters = sda.sigmoid_layers[0].W.get_value()
    print filters.shape
    image = PIL.Image.fromarray(tile_raster_images(
                      X=filters.T,
		    	img_shape=(17, 17), tile_shape=(4, 20),
		    	tile_spacing=(1, 1),scale_rows_to_unit_interval=True))
    image.save('outputs/LLnet_filters.png')
   
#####################################################################################################################
#                                                                                                                   #
#                                                Terminal Commands                                                  #
#                                                                                                                   #
#####################################################################################################################

if __name__ == '__main__':

        print(chr(27) + "[2J")
        
        # Command line interface --------------------
        if len(sys.argv) > 1:
		if len(sys.argv[1])>0:
		    if sys.argv[1]=='train':
		        tr_dataset = str(sys.argv[2])
			test_SdA()
			exit()
		    if sys.argv[1]=='test':
			print '... Runnning algorithm!'
			te_noisy_image = str(sys.argv[2])
			model_to_load = str(sys.argv[3])
			f = file(model_to_load, 'rb')
			sda = cPickle.load(f)
			f.close()
			denoise_overlapped_strides();
			print 'Completed:', te_noisy_image
			exit()
	# -------------------------------------------
        
	msg = "You are currently running the image enhancement program, LLNet, developed by Akintayo, Lore, and Sarkar. What would you like to do?"
	choices = ["Train Model","Enhance Single/Multiple Images","Exit Program"]
	reply = buttonbox(msg, title="Welcome to LLNet!", choices=choices)
	if reply == "Exit Program":
	    exit()
         
        if reply == "Train Model":
        
            if ccbox('You are currently training a new model. The model file might be overwritten. Continue?','Information')==True:
                tr_dataset = fileopenbox(title='Select training data.',default='*',filetypes=["*.mat"])
            	test_SdA()
            else:
                msgbox("Program terminated. Goodbye!")
                exit()
            
        if reply == "Enhance Single/Multiple Images":
            
	    # Present model to load
	    model_to_load = fileopenbox(title='Select your model to load.',default='*',filetypes=["*.obj"])
	    f = file(model_to_load, 'rb')
	    sda = cPickle.load(f)
	    f.close()
	    
            # Load the test image
            te_noisy_image_list = fileopenbox(title='Select an image to enhance. Multiple images are allowed; hold SHIFT and click to select.',default='*',filetypes=["*.png", ["*.jpg", "*.jpeg", "JPEG Files"] , '*.bmp' , '*.gif'  ],multiple=True)
            print te_noisy_image_list
            print '... Runnning algorithm!'

            for f in te_noisy_image_list:
            
                    te_noisy_image = f
		    denoise_overlapped_strides();
		    print 'Completed:', f

