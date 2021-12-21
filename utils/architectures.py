import tensorflow as tf
from utils.ResidualAttentionNetwork import ResidualAttentionNetwork
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Input
from tensorflow.keras.layers import BatchNormalization, Activation, Flatten, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.resnet import ResNet152



def residual_attention_56(input_shape, output_size): #m = 1 so 36*1 + 20 = 56   
    '''
    Attention-56 Network Structure
    '''

    
    ra = ResidualAttentionNetwork(input_shape, output_size, p=1, t=2, r=1, filters = [[16,16,64],[32,32,128],[64,64,256],[128,128,512]])
    
    filters = ra.filters

    input_data = Input(shape=ra.input_shape)  #32x32
    #print(input_data.shape)
    conv1 = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding='same')(input_data)  

    # Residual Attention Module 1
    filters_st1 = filters[0]
    res1 = ra.residualUnit(conv1, filters=filters_st1, type_resunit='in')
    att1 = ra.attentionModule1(res1, filters=filters_st1, learning_mechanism ='ARL')  

    # Residual Attention Module 2
    filters_st2 = filters[1]
    res2 = ra.residualUnit(att1, filters=filters_st2, type_resunit='out')
    att2 = ra.attentionModule2(res2, filters=filters_st2, learning_mechanism='ARL') 

    # Residual Attention Module 3
    filters_st3 = filters[2]
    res3 = ra.residualUnit(att2, filters=filters_st3, type_resunit='out')
    att3 = ra.attentionModule3(res3, filters=filters_st3, learning_mechanism='ARL') 

    filters_ending = filters[3]
    att3 = ra.residualUnit(att3, filters=filters_ending, type_resunit='out') 

    att3 = ra.residualUnit(att3, filters=filters_ending, type_resunit='in')
    att3 = ra.residualUnit(att3, filters=filters_ending, type_resunit='in')

    batchnorm2 = BatchNormalization()(att3)
    activation2 = Activation('relu')(batchnorm2)
    avg_pool = AveragePooling2D(pool_size=(2,2), strides=(1,1), padding='valid')(activation2) 

    flatten = Flatten()(avg_pool)
    output = Dense(ra.output_size, activation='softmax')(flatten)

    model = Model(inputs=input_data, outputs=output)

    return model






def residual_attention_92(input_shape, output_size): #m = 2 so 36*2 + 20 = 92
    '''
    Attention-92 Network Structure
    '''
    
    ra = ResidualAttentionNetwork(input_shape, output_size, p=1, t=2, r=1, filters = [[16,16,64],[32,32,128],[64,64,256], [128,128,512]])
    

    filters = ra.filters

    input_data = Input(shape=ra.input_shape)

    conv1 = Conv2D(filters=64, kernel_size=(3,3), padding='same', strides=(2,2))(input_data)

    # Residual-Attention Module 1
    filters_st1 = filters[0]
    res1 = ra.residualUnit(conv1, filters=filters_st1, type_resunit='in')
    att1 = ra.attentionModule1(res1, filters=filters_st1, learning_mechanism ='ARL')

    # Residual-Attention Module 2
    filters_st2 = filters[1]
    res2 = ra.residualUnit(att1, filters=filters_st2, type_resunit='out')

    att2 = ra.attentionModule2(res2, filters=filters_st2, learning_mechanism='ARL')
    att2 = ra.attentionModule2(att2, filters=filters_st2, learning_mechanism='ARL')

    # Residual-Attention Module 3
    filters_st3 = filters[2]
    res3 = ra.residualUnit(att2, filters=filters_st3, type_resunit='out')
    att3 = ra.attentionModule3(res3, filters=filters_st3, learning_mechanism='ARL')
    att3 = ra.attentionModule3(att3, filters=filters_st3, learning_mechanism='ARL')
    att3 = ra.attentionModule3(att3, filters=filters_st3, learning_mechanism='ARL')

    filters_ending = filters[3]
    att3 = ra.residualUnit(att3, filters=filters_ending, type_resunit='out')

    att3 = ra.residualUnit(att3, filters=filters_ending, type_resunit='in')
    att3 = ra.residualUnit(att3, filters=filters_ending, type_resunit='in')

    batchnorm_2 = BatchNormalization()(att3)
    activation2 = Activation('relu')(batchnorm_2)
    avg_pool = AveragePooling2D(pool_size=(2,2), strides=(1,1), padding='valid')(activation2)

    flatten = Flatten()(avg_pool)
    output = Dense(ra.output_size, activation='softmax')(flatten)

    model = Model(inputs=input_data, outputs=output)

    return model



def resnet_152(output_shape):
    

    model = Sequential()

    model.add(ResNet152(include_top=False, pooling='avg'))
    model.add(Dense(output_shape, activation='softmax'))

    return model
        



