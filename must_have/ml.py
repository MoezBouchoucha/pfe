from matplotlib import pyplot as plt
from numpy import zeros
from numpy import ones
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from tensorflow.keras.layers import Input,LeakyReLU,BatchNormalization,Activation, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout,Reshape
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention


def conv_block(inputs, filters):
    conv1 = Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    conv2 = Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    attn = MultiHeadAttention(num_heads=8, key_dim=32)(conv2, conv2, conv2)
    norm = LayerNormalization()(attn)
    dropout = Dropout(0.5)(norm)
    return dropout


def upconv_block(inputs, skip_inputs, filters):
    upconv = UpSampling2D(size=(2, 2))(inputs)
    conv1 = Conv2D(filters, kernel_size=(2, 2), padding='same', activation='relu')(upconv)
    concat = Concatenate()([conv1, skip_inputs])
    conv2 = Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu')(concat)
    conv3 = Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu')(conv2)
    attn = MultiHeadAttention(num_heads=8, key_dim=32)(conv3, conv3, conv3)
    norm = LayerNormalization()(attn)
    dropout = Dropout(0.5)(norm)
    return dropout


def u_net(input_shape, output_shape):
    inputs = Input(shape=input_shape)

    # Contracting Path
    conv1 = conv_block(inputs, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(pool3, 512)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Bottom
    conv5 = conv_block(pool4, 1024)

    # Expansive Path
    up6 = upconv_block(conv5, skip_inputs=drop4, filters=512)
    up7 = upconv_block(up6, skip_inputs=conv3, filters=256)
    up8 = upconv_block(up7, skip_inputs=conv2, filters=128)
    up9 = upconv_block(up8, skip_inputs=conv1, filters=64)

    outputs = Conv2D(3, kernel_size=(1, 1), activation='sigmoid')(up9)
    reshaped_outputs = Reshape(output_shape)(outputs)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=reshaped_outputs)
    return model






def distance_matrix(array1, array2):
    num_point, num_features = array1.shape
    expanded_array1 = tf.tile(array1, (num_point, 1))
    expanded_array2 = tf.reshape(
            tf.tile(tf.expand_dims(array2, 1), 
                    (1, num_point, 1)),
            (-1, num_features))
    distances = tf.norm(expanded_array1-expanded_array2, axis=1)
    distances = tf.reshape(distances, (num_point, num_point))
    return distances
def av_dist(array1, array2):
    distances = distance_matrix(array1, array2)
    distances = tf.reduce_min(distances, axis=1)
    distances = tf.reduce_mean(distances)
    return distances
def av_dist_sum(arrays):
    array1, array2 = arrays
    av_dist1 = av_dist(array1, array2)
    av_dist2 = av_dist(array2, array1)
    return av_dist1+av_dist2
def chamfer_distance(array1, array2):
    batch_size, num_point, num_features = array1.shape
    dist = tf.reduce_mean(
               tf.map_fn(av_dist_sum, elems=(array1, array2), dtype=tf.float64)
           )
    return dist


class model:
    def __init__(self):
        self.train_losses=[]
        self.valid_losses=[]
        

    #########################################################################################################
    def helper(self):
        print("el class hedhy bech ndefini feha el modele , fonction train for one epoch w class ll generator , class ll descriminator w class ll model kemel mte3 el gan")
        print("mele5er hedhy feha el parametres w modeles li bech traini w kifech traini epoch wa7da bark.")
        print("to do : methode model ,methode generator , methode descriminator , methode train_one_epoch ")

    #########################################################################################################
    def split_train_valid(self,x,y,ratio=0.2):
        nbr_train=len(x)-(len(x)*ratio)
        
        x_train=x[:nbr_train]
        y_train=y[:nbr_train]
        x_valid=x[nbr_train:]
        y_valid=y[nbr_train:]
        
        return x_train,y_train,x_valid,y_valid
    
    #########################################################################################################

    def define_generator(self,in_shape,out_shape):
        return u_net(in_shape,out_shape)
    
    #########################################################################################################
    def define_discriminator(self,obj_shape):
        i = tf.keras.initializers.RandomNormal(stddev=0.02) #As described in the original paper
        
        # source image input
        in_src_image = Input(shape=obj_shape)  #Image we want to convert to another image
        # target image input
        in_target_image = Input(shape=obj_shape)  #Image we want to generate after training. 
        
        # concatenate images, channel-wise
        merged = Concatenate()([in_src_image, in_target_image])
        
        # C64: 4x4 kernel Stride 2x2
        d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer="random_normal")(merged)
        d = LeakyReLU(alpha=0.2)(d)
        # C128: 4x4 kernel Stride 2x2
        d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer="random_normal")(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C256: 4x4 kernel Stride 2x2
        d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer="random_normal")(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C512: 4x4 kernel Stride 2x2 
        d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer="random_normal")(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # second last output layer : 4x4 kernel but Stride 1x1
        d = Conv2D(512, (4,4), padding='same', kernel_initializer="random_normal")(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # patch output
        d = Conv2D(1, (4,4), padding='same', kernel_initializer="random_normal")(d)
        patch_out = Activation('sigmoid')(d)
        # define model
        model = Model([in_src_image, in_target_image], patch_out)
        
        # opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer="adam", loss_weights=[0.5])
        return model


    #########################################################################################################
    def define_gan(self,gen,desc,shape,optimizer):
        for layer in desc.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = False 
        
        in_src = Input(shape=shape)
        
        gen_out = gen(in_src)
        
        dis_out = desc([in_src, gen_out])
        
        model = Model(in_src, [dis_out, gen_out])
        
        opt = Adam(lr=0.0002, beta_1=0.5)
        
        #normalment moch kima el pix2pix bech ne5dem b el mae 3ala eses metric el thenya ama bech ne5dem b chamfer distance 
        model.compile(loss=['binary_crossentropy', 'mae'], 
                optimizer=optimizer, loss_weights=[70,30])
        return model
    
    #########################################################################################################
    def callback(self,generator,data):
        print("generator,data")
    
    #########################################################################################################
    
    def gen_real(self,x , y,patch_shape):
        z = ones((len(x), patch_shape, patch_shape, 1))
        return x,y,z

    #########################################################################################################
    
    def gen_fake(self,g_model, x,patch_shape):
        y = g_model.predict(x)
        z = zeros((len(y), patch_shape, patch_shape, 1))
        return x,y,z
    
    #########################################################################################################
    def perform_validation(self,x,y,model):
        pass
    #########################################################################################################
    
    def train_for_one_epoch(self,x_train,y_train,x_valid,y_valid,g_model,d_model,gan_model,num_epochs,batch_size,patch_value):
        #generati real samples 
        x,y,z=self.gen_real(x_train,y_train,patch_shape=patch_value)

        #generati fake samples 
        x_fake,y_fake,z_fake=self.gen_fake(g_model,x,patch_shape=patch_value)
		
        # discriminator fi el real 
        d_loss1 = d_model.train_on_batch([x, y], z)
		
        # generator fi el real
        d_loss2 = d_model.train_on_batch([x_fake, y_fake], z_fake)
		
        # generator
        g_loss, _, _ = gan_model.train_on_batch(x, [z,y])
        
        # self.callback() # type: ignore hedhy el vs code zedetha -_- 
        
        # summarize performance
        print('real samples: d1[%.3f]  fake samples : d2[%.3f]  gan loss : g[%.3f]    <===>' % ( d_loss1, d_loss2, g_loss))
