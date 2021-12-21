from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, AveragePooling2D, Input, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization, Activation, Add, Multiply, Flatten, Dense, Dropout
from tensorflow.keras.applications.resnet import ResNet152

class ResidualAttentionNetwork():

    def __init__(self, input_shape, output_size, p=1, t=2, r=1,filters = [[16,16,64],[32,32,128],[64,64,256],[128,128,512]]):
        self.input_shape = input_shape #input shape of the image (32*32*3)
        self.output_size = output_size #number of classes (10 for cifar10 and 100 for cifar100)
        self.p = p #total number of residual units in each stage
        self.t = t #total number of residual units in trunk branch
        self.r = r #total number of residual units in soft mask branch
        self.filters = filters #sizes of filters during each stage
        
    def residualUnit(self, res_ip, filters, type_resunit='in'):
        '''
        Defining a residual unit used for the RAN
        '''
        # Assigning filter sizes to each filter
        filter1 = filters[0]
        filter2 = filters[1]
        filter3 = filters[2]

        #Initializing identity mapping 
        identityx = res_ip
        
        #1x1 filters help in reducing and restoring dimensions
        batchnorm1 = BatchNormalization()(res_ip)
        activation1 = Activation('relu')(batchnorm1)
        conv1 = Conv2D(filters = filter1, kernel_size = (1,1), padding = 'same')(activation1)
        
        #3x3 layer acting as a bottleneck
        batchnorm2 = BatchNormalization()(conv1)
        activation2 = Activation('relu')(batchnorm2)
        
        #residual unit type definition
        if type_resunit == 'in':
            conv2 = Conv2D(filters = filter2, kernel_size = (3,3), padding = 'same', strides = (1,1))(activation2)
        else: 
            conv2 = Conv2D(filters = filter2, kernel_size = (3,3), padding = 'same', strides = (2,2))(activation2)

        #1x1 filters help in reducing and restoring dimensions
        batchnorm3 = BatchNormalization()(conv2)
        activation3 = Activation('relu')(batchnorm3)
        conv3 = Conv2D(filters=filter3, kernel_size=(1,1), padding='same')(activation3)

        #Identity mapping according to type of the residual unit
        if identityx.shape != conv3.shape:
            filter_c = conv3.shape[-1]  # number of filters
            if type_resunit == 'in':
                identityx = Conv2D(filters = filter_c, kernel_size = (1,1), padding = 'same', strides = (1,1))(identityx) 
            else:  
                identityx = Conv2D(filters = filter_c, kernel_size = (3,3), padding = 'same', strides = (2,2))(identityx) 

        output = Add()([identityx, conv3])
        
        return output
    
    
    def attentionModule1(self, input_unit, filters, learn_mech):
        
        p = self.p   # Number of preprocessing residual units
        
        # Adding p number of residual units
        for i in range(p):
            am_unit = self.residualUnit(input_unit, filters, type_resunit='in')
        
        t = self.t   # Number of residual units in the trunk branch

        # Adding t number of residual units
        for i in range(t):
            trunk_unit = self.residualUnit(am_unit, filters, type_resunit='in')
        
        # Downsampling
        down1 = MaxPool2D(pool_size = (3,3), padding = 'same', strides = (2,2))(am_unit)
        
        r = self.r # Number of Residual Units between adjacent pooling layer in the mask branch
        
        # Adding r number of residual units
        for i in range(r):
            down1 = self.residualUnit(down1, filters, type_resunit='in')
        
        # adding a skip connection
        skip1 = self.residualUnit(down1, filters, type_resunit = 'in')
        
        # Downsampling
        down2 = MaxPool2D(pool_size=(3,3), padding = 'same', strides = (2,2))(down1)
        
        # Adding r number of residual units
        for i in range(self.r):
            down2 = self.residualUnit(down2, filters, type_resunit = 'in')
        
        # adding a skip connection
        skip2 = self.residualUnit(down2, filters, type_resunit= 'in')
        
        down3 = MaxPool2D(pool_size = (3,3), padding = 'same', strides = (2,2))(down2)
        
        # Adding 2r number of residual units
        for i in range(self.r * 2):
            down3 = self.residualUnit(down3, filters, type_resunit='in')
        
        # Upsampling
        up1 = UpSampling2D(size=(2,2))(down3) 
        
        addunit1 = Add()([up1, skip2])
        
        # Adding r number of residual units
        for i in range(self.r):
            addunit1 = self.residualUnit(addunit1, filters, type_resunit='in')
        
        # Upsampling
        up2 = UpSampling2D(size=(2,2))(addunit1) 
        
        addunit2 = Add()([up2, skip1])
        
        # Adding r number of residual units
        for i in range(self.r):
            addunit2 = self.residualUnit(addunit2, filters, type_resunit = 'in')
        up3 = UpSampling2D(size=(2,2))(addunit2) 
        
        convFilt = up3.shape[-1]
        conv1 = Conv2D(filters = convFilt, kernel_size = (1,1), padding = 'same')(up3)
        conv2 = Conv2D(filters = convFilt, kernel_size = (1,1), padding = 'same')(conv1)


        softmask = Activation('sigmoid')(conv2)
        
        # Naive Attention Learning or Attention Residual Learning
        if learn_mech == 'NAL':
            output = Multiply()([trunk_unit, softmask])
        else:
            output = Multiply()([trunk_unit, softmask])
            output = Add()([output, trunk_unit])
        
        # terminal p block for the module
        for i in range(self.p):
            output = self.residualUnit(output, filters)
        
        return output
        
    
    def attentionModule2(self, input_unit, filters, learn_mech):
        
        p = self.p
        # Adding p number of residual units
        for i in range(self.p):
            am_unit = self.residualUnit(input_unit, filters, type_resunit = 'in')
        
        t = self.t
    
        # Adding t number of residual units
        for i in range(self.t):
            trunk_unit = self.residualUnit(am_unit, filters, type_resunit = 'in')
        
        # Downsampling
        down1 = MaxPool2D(pool_size = (3,3), padding = 'same', strides = (2,2))(am_unit)
        
        r = self.r
        # Adding r number of residual units
        for i in range(r):
            down1 = self.residualUnit(down1, filters, type_resunit='in')

        # Skip connection
        skip1 = self.residualUnit(down1, filters, type_resunit='in')
        
        # Downsampling
        down2 = MaxPool2D(pool_size = (3,3), padding = 'same', strides = (2,2))(down1)
        
        # Adding 2r number of residual units
        for i in range(r*2):
            down2 = self.residualUnit(down2, filters, type_resunit='in')
        
        # Upsampling
        up1 = UpSampling2D(size=(2,2))(down2) 

        addunit2 = Add()([up1, skip1])
        
        # Adding r number of residual units
        for i in range(self.r):
            addunit2 = self.residualUnit(addunit2, filters, type_resunit='in')
        
        # Upsampling
        up3 = UpSampling2D(size=(2,2))(addunit2) 
        
        convFilt = up3.shape[-1]
        conv1 = Conv2D(filters = convFilt, kernel_size = (1,1), padding = 'same')(up3)
        conv2 = Conv2D(filters = convFilt, kernel_size = (1,1), padding = 'same')(conv1)
        softmask = Activation('sigmoid')(conv2)
        
        # Naive Attention Learning or Attention Residual Learning
        if learn_mech == 'NAL':
            output = Multiply()([trunk_unit, softmask])
        else:
            output = Multiply()([trunk_unit, softmask])
            output = Add()([output, trunk_unit])
        
        # terminal p block for the module
        for i in range(self.p):
            output = self.residualUnit(output, filters)
            
        return output
        
        
    def attentionModule3(self, input_unit, filters, learn_mech):
        
        p = self.p
        # Adding p number of residual units
        for i in range(p):
            am_unit = self.residualUnit(input_unit, filters, type_resunit='in')
        
        t = self.t
        # Adding t number of residual units
        for i in range(t):
            trunk_unit = self.residualUnit(am_unit, filters, type_resunit='in')
        
        # Downsampling
        down1 = MaxPool2D(pool_size=(3,3), padding='same', strides=(2,2))(am_unit)
        
        r = self.r
        # Adding r number of residual units
        for i in range(r):
            down1 = self.residualUnit(down1, filters, type_resunit='in')
        
        # Upsampling
        up3 = UpSampling2D(size=(2,2))(down1) 
        
        convFilt = up3.shape[-1]
        conv1 = Conv2D(filters = convFilt, kernel_size = (1,1), padding = 'same')(up3)
        conv2 = Conv2D(filters = convFilt, kernel_size = (1,1), padding = 'same')(conv1)
        softmask = Activation('sigmoid')(conv2)
        
        # Naive Attention Learning or Attention Residual Learning
        if learn_mech == 'NAL':
            output = Multiply()([trunk_unit, softmask])
        else:
            output = Multiply()([trunk_unit, softmask])
            output = Add()([output, trunk_unit])
        
        # terminal p block for the module
        for i in range(p):
            output = self.residualUnit(output, filters)
            
        return output
    
    
        
